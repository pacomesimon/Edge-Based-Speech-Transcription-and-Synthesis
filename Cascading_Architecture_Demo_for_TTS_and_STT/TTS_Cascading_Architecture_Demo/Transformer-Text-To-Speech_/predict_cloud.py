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


model = TransformerTTS(
    vocab_size=len(pickle.load(open('input/char_to_idx.pickle', 'rb')))+1,
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.backends.cudnn.benchmark = True
# device = torch.device('cpu')
model = model.to(device)

def inference_fn_cloud(encoder_embed, decoder_embed):
    if os.path.exists(config.checkpoint):
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("model checkpoint loaded.")
    running_loss = 0
    model.eval()
    with torch.no_grad():
          mel_spect_post_pred, mel_spect_pred, end_logits_pred = model(None, None, None, is_from_edge = True, 
                                                                        x_embed = encoder_embed, y_embed = decoder_embed)
    return mel_spect_post_pred, mel_spect_pred, end_logits_pred


