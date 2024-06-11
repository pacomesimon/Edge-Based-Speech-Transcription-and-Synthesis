from flask import Flask, request
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
model_server = SpeechT5ForTextToSpeech.from_pretrained("NMutangana/speecht5_tts_common_voice_swahili")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
torch.set_printoptions(threshold=torch.inf)
tensor = torch.tensor
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!, This is Pacome!'

@app.route('/receiveParameters')
def receive_parameters():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output =  dict(**eval(text))
    print("::DATA::"*4)
    print("model_output:",type(model_output))
    return repr(model_output), 200
    
@app.route('/receiveTextData', methods=['POST'])
def receive_text_data():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    return {'received_text': text}, 200

@app.route('/encoder', methods=['POST'])
def encode_encoder_output():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = model_server.speecht5.encoder.wrapped_encoder(
            **eval(text)
    )
    print("::ENC::"*4)
    print("model_output:",type(model_output))
    return repr(model_output), 200

@app.route('/decoder', methods=['POST'])
def encode_decoder_output():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = model_server.speecht5.decoder.wrapped_decoder(
            **eval(text)
    )
    print("::DEC::"*4)
    print("model_output:",type(model_output))
    return repr(model_output), 200

@app.route('/decoder_full', methods=['POST'])
def encode_decoder_full_output():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = model_server.speecht5.decoder(
            **eval(text)
    )
    print("::DEC_F"*4)
    print("model_output:",type(model_output))
    return repr(model_output), 200

@app.route('/vocoder', methods=['POST'])
def vocode_from_spectrogram():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = vocoder(
            **eval(text)
    )
    print("::VOC::"*4)
    print("model_output:",type(model_output))
    return repr(model_output), 200
app.run()   