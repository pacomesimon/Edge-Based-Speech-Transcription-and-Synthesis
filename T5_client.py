import torch
import re
import numpy as np

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan

from transformers import modeling_outputs
import soundfile as sf
import requests
import os

SERVER_URL = 'http://127.0.0.1:5000/'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("NMutangana/speecht5_tts_common_voice_swahili")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


if not os.path.exists('data.json'):
    os.system('curl -X GET "https://datasets-server.huggingface.co/rows?dataset=Matthijs%2Fcmu-arctic-xvectors&config=default&split=validation&offset=0&length=100" > data.json')

text = "Moja mbili tatu nne tano sita saba nane tisa kumi"
print(
    "input text length:", len(text), "maximum length (model): 982"
)
my_input_ids = processor.tokenizer(text, return_tensors="pt")['input_ids'].to(device)

# prompt: load the data.json in a dictionary

import json
with open('data.json') as f:
    speakers_embeddings_data = json.load(f)


sample_speakers_embeddings = torch.tensor([
    # "put your 512-long embeddings here"
    speakers_embeddings_data["rows"][0]['row']['xvector']
 ])

torch.set_printoptions(threshold=torch.inf)
tensor = torch.tensor
BaseModelOutput = modeling_outputs.BaseModelOutput
BaseModelOutputWithPastAndCrossAttentions = modeling_outputs.BaseModelOutputWithPastAndCrossAttentions


def model_speecht5_encoder_wrapped_encoder_forward(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
            ):
    args_dict = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict
    }
    args_dict_text = repr(args_dict)
    response = requests.post(SERVER_URL+'encoder', data=args_dict_text.encode('utf-8'))

    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    return eval(clean_text)
def model_speecht5_decoder_wrapped_decoder_forward(
            hidden_states  = None,
            attention_mask  = None,
            encoder_hidden_states = None,
            encoder_attention_mask= None,
            head_mask = None,
            cross_attn_head_mask = None,
            past_key_values = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            ):
    args_dict = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict
    }
    args_dict_text = repr(args_dict)
    response = requests.post(SERVER_URL+'decoder', data=args_dict_text.encode('utf-8'))
    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    return eval(clean_text)
def model_speecht5_decoder_forward(
            hidden_states  = None,
            attention_mask  = None,
            encoder_hidden_states = None,
            encoder_attention_mask= None,
            head_mask = None,
            cross_attn_head_mask = None,
            past_key_values = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            ):
    args_dict = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": return_dict
    }
    args_dict_text = repr(args_dict)
    response = requests.post(SERVER_URL+'decoder_full', data=args_dict_text.encode('utf-8'))
    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    # print("*-_*-_"*8,"DECoding Done!!!")
    return eval(clean_text)

def vocoder_forward(spectrogram = None):
    args_dict = {
        "spectrogram": spectrogram
    }
    args_dict_text = repr(args_dict)
    response = requests.post(SERVER_URL+'vocoder', data=args_dict_text.encode('utf-8'))
    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    return eval(clean_text)

model.speecht5.encoder.wrapped_encoder.forward = model_speecht5_encoder_wrapped_encoder_forward
vocoder.forward = vocoder_forward

spectrogram = model.generate_speech(my_input_ids, sample_speakers_embeddings)
with torch.no_grad():
    speech = vocoder(spectrogram)

# Save speech to disk as 'generated_speech.wav'
sf.write('./data/generated_speech.wav', speech.squeeze().cpu().numpy(), 16000)
print("Generated speech saved as './data/generated_speech.wav'")