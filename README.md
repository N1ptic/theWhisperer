# 🎙️ WhispererAI

An intelligent voice-based AI assistant that transcribes speech and answers questions in real-time using OpenAI's Whisper and Llama models.

## 🌟 Features

- Real-time audio recording and transcription
- Local speech recognition using Whisper Base model
- AI-powered responses using Llama through Ollama
- High-quality audio processing with noise filtering
- CUDA acceleration support for faster processing
- Cross-platform support (Windows, Linux, macOS)

## 🛠️ Technologies

- Python 3.8+
- OpenAI Whisper (Base model)
- Llama (via Ollama)
- PyTorch
- Transformers
- FFMPEG for audio capture
- SoundFile for audio processing

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- FFMPEG installed on your system
- Ollama installed and running locally
- Compatible audio input device (default or HyperX Cloud Stinger Core Wireless)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WhispererAI.git
cd WhispererAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama from [ollama.ai](https://ollama.ai) and start the service

## 💻 Usage

1. Start the application:
```bash
python app.py
```

2. Use the following controls:
- Press `R` to start recording
- Press `S` to stop recording and process the audio
- Press `C` to clear the screen
- Press `Q` to quit the application

## ⚙️ Configuration

The application uses the following default settings:
- Audio sample rate: 48kHz
- Audio channels: Mono
- Whisper Model: Base
- LLM: Llama (via Ollama)
- Device: CUDA if available, CPU otherwise
- Audio filters: High-pass (50Hz), Low-pass (15kHz), Volume boost (1.5x)

## 🎤 Audio Device Configuration

- Windows: Automatically detects HyperX Cloud Stinger Core Wireless DTS
- Linux/macOS: Uses default audio input device
- Lists available audio devices if preferred device is not found

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Notes

- Ensure Ollama is running before starting the application
- Configure your audio input device if the default is not suitable
- For optimal performance, use a CUDA-capable GPU

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
