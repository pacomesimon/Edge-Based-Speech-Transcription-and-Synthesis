import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.io import wavfile
import config
import engine
import librosa
import sys
import pickle
import os
import gc
import transformers
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from librosa.feature import melspectrogram
from Transformer_tts_model.TransformerTTSModel import TransformerTTS
from tqdm import tqdm 

class TransformerLoaderInference(torch.utils.data.Dataset):
    def __init__(self, text_data, mel_transforms=None, normalize=False):
        self.text_data = text_data
        self.transforms = mel_transforms
        self.normalize = normalize
        self.char_to_idx = pickle.load(open('input/char_to_idx.pickle', 'rb'))
    
    def __len__(self):
        return len(self.text_data)

    def data_preprocess_(self, text):
        char_idx = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        return char_idx
    

    def normalize_(self, mel):
        #Normalizing data between -4 and 4 
        #Converges even more faster
        mel = np.clip(
            (config.scaling_factor)*((mel - config.min_db_level)/-config.min_db_level) - config.scaling_factor, 
            -config.scaling_factor, config.scaling_factor
        )
        return mel

    
    def __getitem__(self, idx):
        # print("idx:",idx)
        text = self.text_data[idx]
        text_idx = self.data_preprocess_(text)

        audio_file = np.array([0]).astype(np.float32)

        audio_file, _ = librosa.effects.trim(audio_file)
        mel_spect = melspectrogram(
            y = audio_file,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            win_length=config.win_length
        )
        
        pre_mel_spect = np.zeros((1, config.n_mels))
        mel_spect = (mel_spect).T #librosa.power_to_db(mel_spect).T
        mel_spect = np.concatenate((pre_mel_spect, mel_spect), axis=0)

        # if self.normalize:
        #     mel_spect = self.normalize_(mel_spect)

        mel_spect = torch.tensor(mel_spect, dtype=torch.float)
        mel_mask = [1]*mel_spect.shape[0]

        end_logits = [0]*(len(mel_spect) - 1)
        end_logits += [1]

        # if self.transforms:
        #     for transform in self.transforms:
        #         if np.random.randint(0, 11) == 10:
        #             mel_spect = transform(mel_spect).squeeze(0)

        return {
            'original_text'  : text,
            'mel_spect'     : mel_spect,
            'mel_mask'      : torch.tensor(mel_mask, dtype=torch.long),
            'text_idx'      : torch.tensor(text_idx, dtype=torch.long),
            'end_logits'    : torch.tensor(end_logits, dtype=torch.float),
        }

class MyCollate:
    def __init__(self, pad_idx, spect_pad):
        self.pad_idx = pad_idx
        self.spect_pad =spect_pad
    
    def __call__(self, batch):
        text_idx = [item['text_idx'] for item in batch]
        padded_text_idx = pad_sequence(
            text_idx,
            batch_first=True,
            padding_value=self.pad_idx
        )
        end_logits = [item['end_logits'] for item in batch]
        padded_end_logits  = pad_sequence(
            end_logits,
            batch_first=True,
            padding_value=0
        )
        original_text = [item['original_text'] for item in batch]
        mel_mask = [item['mel_mask'] for item in batch]
        padded_mel_mask = pad_sequence(
            mel_mask,
            batch_first=True,
            padding_value=0
        )
        mel_spects = [item['mel_spect'] for item in batch]

        batch_size, max_len = padded_mel_mask.shape

        padded_mel_spect = torch.zeros(batch_size, max_len, mel_spects[0].shape[-1])

        for num,mel_spect in enumerate(mel_spects):
            padded_mel_spect[num, :mel_spect.shape[0]] = mel_spect
        
        return {
            'original_text'  : original_text,
            'mel_spect'     : padded_mel_spect,
            'mel_mask'      : padded_mel_mask,
            'text_idx'      : padded_text_idx,
            'end_logits'    : padded_end_logits
        }


def inference_fn(model, data_loader, device):
    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("model checkpoint loaded.")
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for num, data in (enumerate(data_loader)):
            end_logits = data['end_logits'].to(device)
            mel_spect = data['mel_spect'].to(device)
            print("mel_spect[:, :-1]:",mel_spect[:, :-1].shape)
            text_idx = data['text_idx'].to(device)
            text = data['original_text']
            mel_mask = data['mel_mask'].to(device)
            # print("mel_mask.shape:",mel_mask)
            # On edge device
            encoder_out = model.get_encoder_output(text_idx)
            # On server
            mel_spect_post_pred, mel_spect_pred, end_logits_pred = model.get_decoder_output(encoder_out, mel_spect[:, :-1], mel_mask[:, :-1])
            initial_overhead_length = mel_spect.shape[1]
            payload_length = (config.sample_rate//200) * 3 # 3-second(s) long output
            bs,sq_len, fts = mel_spect.shape
            mel_spect = torch.concatenate((mel_spect, torch.zeros((bs,payload_length, fts))), axis=1)
            mel_mask = torch.tensor([[1]*(initial_overhead_length+payload_length)], dtype=torch.long)
            # print("melshapes:", mel_spect.shape, mel_mask.shape)
            for _ in tqdm(range(payload_length)):
              # print("here we go, _ is:",(initial_overhead_length+_+(-1024)),(initial_overhead_length+_))
              # print("initial_overhead_length+_:",mel_spect[:,0:(initial_overhead_length+_),:].shape, 
              #                                                                                               mel_mask[:,(initial_overhead_length+_+(-1024)):(initial_overhead_length+_)].shape)
              mel_spect[:,initial_overhead_length+_:,:] = mel_spect_pred[:,-1:,:]
              if (initial_overhead_length+_)<1024:
                mel_spect_post_pred, mel_spect_pred, end_logits_pred = model.get_decoder_output(encoder_out, mel_spect[:,:(initial_overhead_length+_),:], 
                                                                                                              mel_mask[:,:(initial_overhead_length+_)])
              else:
                mel_spect_post_pred, mel_spect_pred, end_logits_pred = model.get_decoder_output(encoder_out, mel_spect[:,(initial_overhead_length+_+(-1024)):(initial_overhead_length+_),:], 
                                                                                                            mel_mask[:,(initial_overhead_length+_+(-1024)):(initial_overhead_length+_)])
    print("mel_spect.shape:",mel_spect.shape)
    audio_signal = librosa.feature.inverse.mel_to_audio(
    mel_spect[0].T.cpu().numpy(),
    sr=config.sample_rate,
    n_fft=config.win_length,
    hop_length=config.hop_length,
    win_length=config.win_length,
    window='hann'
    )
    print("audio_signal:",audio_signal)
    print("audio_signal.shape:",audio_signal.shape)
    wavfile.write("inference.wav", config.sample_rate, audio_signal.astype(np.float32))
    return audio_signal


# print("Enter text:")
# text = input()
if len(sys.argv) > 1:
        arg1 = sys.argv[1]  # First argument
        # arg2 = sys.argv[2]  # Second argument (if provided)
        # ... process the arguments as needed ...
else:
        print("No arguments provided.")
print("Input text:",arg1)
text = arg1
data = TransformerLoaderInference(text_data=[text])
vocab_size = len(data.char_to_idx) + 1
pad_idx = 0


data_loader = torch.utils.data.DataLoader(
    data,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    collate_fn=MyCollate(
        pad_idx=pad_idx, 
        spect_pad=-config.scaling_factor
    )
)

model = TransformerTTS(
    vocab_size=vocab_size,
    embed_dims=config.embed_dims,
    hidden_dims=config.hidden_dims, 
    heads=config.heads,
    forward_expansion=config.forward_expansion,
    num_layers=config.num_layers,
    dropout=config.dropout,
    mel_dims=config.n_mels,
    max_len=config.max_len,
    pad_idx=config.pad_idx
)
# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# torch.backends.cudnn.benchmark = True
device = torch.device('cpu')
model = model.to(device)

# print("data:",data)
# print("mel_shape:",data["mel_spect"].shape)
inference_fn(model, data_loader, device)

