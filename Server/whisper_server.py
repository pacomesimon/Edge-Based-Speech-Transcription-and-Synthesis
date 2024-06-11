from flask import Flask, request
from transformers import WhisperForConditionalGeneration
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model_server = WhisperForConditionalGeneration.from_pretrained("NMutangana/whisper-small-rw").to(device)
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

@app.route('/encoder_full', methods=['POST'])
def encode_full():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = whisper_model_server.model.encoder(
            **eval(text)
    )
    print(f"::ENC::"*2)
    print("model_output:",type(model_output))
    return repr(model_output), 200


@app.route('/decoder_full', methods=['POST'])
def decode_full():
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = whisper_model_server.model.decoder(
            **eval(text)
    )
    print(f"::DEC::"*2)
    print("model_output:",type(model_output))
    return repr(model_output), 200

@app.route('/encoder', methods=['POST'])
def encode_encoder_output():
    layer_idx = int(request.args.get('layer_idx'))
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = whisper_model_server.model.encoder.layers[layer_idx](
            **eval(text)
    )
    print(f"::ENC::layer_{layer_idx}"*2)
    print("model_output:",type(model_output))
    return repr(model_output), 200

@app.route('/decoder', methods=['POST'])
def encode_decoder_output():
    layer_idx = int(request.args.get('layer_idx'))
    text = request.data.decode('utf-8')  # Decode the raw bytes to text
    model_output = whisper_model_server.model.decoder.layers[layer_idx](
            **eval(text)
    )
    print(f"::DEC::layer_{layer_idx}"*2)
    print("model_output:",type(model_output))
    return repr(model_output), 200

app.run()
    