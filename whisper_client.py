import torch
import torch.nn as nn
import re
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# from transformers import pipeline
# from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import types
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import modeling_outputs
import requests

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=None, task="transcribe")
whisper_model = WhisperForConditionalGeneration.from_pretrained("NMutangana/whisper-small-swahili").to(device)
whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language=None, task="transcribe")

torch.set_printoptions(threshold=torch.inf)
tensor = torch.tensor
BaseModelOutput = modeling_outputs.BaseModelOutput
BaseModelOutputWithPastAndCrossAttentions = modeling_outputs.BaseModelOutputWithPastAndCrossAttentions


SERVER_URL = 'http://127.0.1.0:5000/'

def whisper_model_model_encoder_forward(
        input_features=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
    args_dict = {
          "input_features": input_features,
          "attention_mask": attention_mask,
          "head_mask": head_mask,
          "output_attentions": output_attentions,
          "output_hidden_states": output_hidden_states,
          "return_dict": return_dict
      }
    args_dict_text = repr(args_dict)

    response = requests.post(SERVER_URL+'encoder_full',
                             data=args_dict_text.encode('utf-8')
                             )

    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    MY_HTML_RESPONSES += "<br>"
    return eval(clean_text)

def whisper_model_model_decoder_forward(
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
    args_dict = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
          "encoder_hidden_states": encoder_hidden_states,
          "head_mask": head_mask,
          "cross_attn_head_mask": cross_attn_head_mask,
          "past_key_values": past_key_values,
          "inputs_embeds": inputs_embeds,
          "position_ids": position_ids,
          "use_cache": use_cache,
          "output_attentions": output_attentions,
          "output_hidden_states": output_hidden_states,
          "return_dict": return_dict

      }
    args_dict_text = repr(args_dict)

    response = requests.post(SERVER_URL+'decoder_full',
                             data=args_dict_text.encode('utf-8')
                             )

    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    MY_HTML_RESPONSES += "<br>"
    return eval(clean_text)

def whisper_model_model_encoder_layers_layer_forward(
        self,
        hidden_states= None,
        attention_mask= None,
        layer_head_mask = None,
        output_attentions = False,
        ):
    args_dict = {
          "hidden_states": hidden_states,
          "attention_mask": attention_mask,
          "layer_head_mask": layer_head_mask,
          "output_attentions": output_attentions
      }
    args_dict_text = repr(args_dict)

    response = requests.post(SERVER_URL+'encoder',
                             data=args_dict_text.encode('utf-8'),
                             params = {'layer_idx': self.idx}
                             )

    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    MY_HTML_RESPONSES += "<br>"
    return eval(clean_text)

def whisper_model_model_decoder_layers_layer_forward(
        self,
        hidden_states = None,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask  = None,
        layer_head_mask = None,
        cross_attn_layer_head_mask = None,
        past_key_value  = None,
        output_attentions  = False,
        use_cache  = True,
        ):
    args_dict = {
          "hidden_states": hidden_states,
          "attention_mask": attention_mask,
          "encoder_hidden_states": encoder_hidden_states,
          "encoder_attention_mask": encoder_attention_mask,
          "layer_head_mask": layer_head_mask,
          "cross_attn_layer_head_mask": cross_attn_layer_head_mask,
          "past_key_value": past_key_value,
          "output_attentions": output_attentions,
          "use_cache": use_cache
      }
    args_dict_text = repr(args_dict)

    response = requests.post(SERVER_URL+'decoder',
                             data=args_dict_text.encode('utf-8'),
                             params = {'layer_idx': self.idx}
                             )

    clean_text = re.sub(r'grad_fn=[^>]*>', '', response.text)
    global MY_HTML_RESPONSES
    MY_HTML_RESPONSES = clean_text
    MY_HTML_RESPONSES += "<br>"
    return eval(clean_text)

whisper_model.model.encoder.forward = whisper_model_model_encoder_forward


def transcribe(audio):
    """
    Code adapted from: https://github.com/huggingface/transformers/issues/21809
    - Check also: https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor?usp=sharing#scrollTo=i5sKbZpsY9-J
    - EASIEST ONE: https://huggingface.co/docs/transformers/model_doc/whisper#inference

    """
    inputs = whisper_processor.feature_extractor(audio, return_tensors="pt", sampling_rate=16_000).input_features.to(device)
    generate_ids = whisper_model.generate(inputs, max_length=480_000, language="<|tr|>", task="transcribe", return_timestamps=True)
    # The function generate() is defined here: https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/models/whisper/generation_whisper.py#L259
    # IT'S VERY DIFFICULT TO UNDERSTAND !!!
    transcription = whisper_processor.tokenizer.decode(generate_ids[0], decode_with_timestamps=True, output_offsets=True)

    return transcription


# Load an audio file as a floating point time series
audio_path = './data/input_audio.wav'
audio, sample_rate = librosa.load(audio_path, sr=16000)

transcription = transcribe(audio)
print(":::"*8)
print("Full transcription:")
print(transcription['offsets'][0]['text'])
