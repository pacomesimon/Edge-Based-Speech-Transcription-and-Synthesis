# [Edge-Based Speech Transcription and Synthesis for Kinyarwanda and Swahili Languages](https://arxiv.org/abs/2510.16497)

This repository contains scripts to demo Flask-enabled TTS (Text-to-Speech) and STT (Speech-to-Text) models for the Swahili language, using [SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5) and [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) models from Hugging Face, finetuned as demonstrated in the [./finetuning_notebooks folder](./finetuning_notebooks)

---

## Contents

- `T5_server.py`: SpeechT5 TTS server (Flask)  
- `T5_client.py`: SpeechT5 TTS client  
- `whisper_server.py`: Whisper STT server (Flask)  
- `whisper_client.py`: Whisper STT client  

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Demos

### 1. Speech-to-Text: Whisper Demo

#### Step 1: Launch the Whisper Server

In one terminal, run:

```bash
python whisper_server.py
```

- This will launch a Flask server serving the Swahili Whisper model (localhost:5000).

#### Step 2: Run the Whisper Client

In a **separate** terminal, ensure you have an audio file at `./data/input_audio.wav` (16kHz WAV).  
Then run:

```bash
python whisper_client.py
```

- This will send the audio to the server and print the transcription for the first segment.

---

### 2. Text-to-Speech: SpeechT5 Demo

#### Step 1: Launch the T5 Server

In one terminal, run:

```bash
python T5_server.py
```

- This starts a Flask server providing SpeechT5 TTS inference on Swahili.

#### Step 2: Run the T5 Client

In a **separate** terminal, run:

```bash
python T5_client.py
```

- This will generate speech audio from a sample Swahili sentence and save the output to `./data/generated_speech.wav`.

---

## Notes

- Make sure to run the servers and clients in different terminals (or using background jobs/tmux).
- The T5 and Whisper servers both run on `localhost:5000` by default; only one server should run at a time **per port**.
- If you wish to use both T5 and Whisper servers at the same time, edit one of the server scripts to change the port number.
- Ensure you have sufficient disk space and memory for the models to download.

---

## Data Files

- Place your input audio at `./data/input_audio.wav` (16kHz, mono, WAV) for the Whisper demo.
- The T5 client will save output as `./data/generated_speech.wav`.

---

## Acknowledgments
- [Edge-Based Speech Transcription and Synthesis for Kinyarwanda and Swahili Languages](https://arxiv.org/abs/2510.16497)

---
