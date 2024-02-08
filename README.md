# LingoNaut for Python

![LingoNaut](header.jpg)

Welcome to LingoNaut! This repository contains a very simple python script for creating a custom speech-to-speech multilingual language learning assistant.

LingoNaut uses OpenAI's Whisper for speech-to-text, any Ollama model of your choice for the LLM, and the TTS package for text-to-speech.

## Installation
1. Make sure you have [ffmpeg](https://ffmpeg.org/download.html) installed on your system with the location of the `bin/` folder added to your path.
2. Install [Ollama](https://ollama.ai/) on your system. **Windows users will need to serve Ollama from WSL**, but can then run client scripts from Powershell.
3. Install conda or miniconda on your system.
4. Navigate to this repo, and use `conda env create -f environment.yml` to install.
5. Use `conda activate lingonaut` to activate your environment.
6. Run `python create_lingonaut_ollama.py` to create the custom model.
7. Run `python lingonaut.py` to launch the session with the language assistant.

## Usage Instructions
- There are no special options for which language to learn. All models used are fully multilingual, simply state your intention and let the assistant guide you.
- After running the Python script, you will see a message in the terminal that says "Awaiting user input..." when it is your move.
  - To ask the assistant questions in English, hold down `Ctrl` and ask your question. On key release, your message will be passed to assistant.
  - **When practicing another language, hold down `SHIFT`** to use a larger version of the Whisper model which is more accurate in non-English transcription.
