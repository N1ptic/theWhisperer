# 🎙️ WhispererAI

An intelligent voice-based AI assistant that transcribes speech and answers questions in real-time using OpenAI's Whisper and GPT models.

## 🌟 Features

- Real-time audio recording and transcription
- Automatic question detection
- AI-powered responses using GPT-3.5 Turbo
- Support for high-quality audio processing
- CUDA acceleration support for faster processing

## 🛠️ Technologies

- Python 3.7+
- OpenAI Whisper (Large v3 model)
- OpenAI GPT-3.5 Turbo
- PyTorch
- Transformers
- FFMPEG for audio capture

## 📋 Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, but recommended)
- FFMPEG installed on your system
- OpenAI API key
- Compatible audio input device

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WhispererAI.git
cd WhispererAI
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 💻 Usage

1. Start the application:
```bash
python app.py
```

2. Use the following controls:
- Press `+` to start recording
- Press `-` to stop recording and process the audio
- Press `q` to quit the application

## ⚙️ Configuration

The application uses the following default settings:
- Audio sample rate: 16kHz
- Audio channels: Mono
- Model: Whisper Large v3
- Device: CUDA if available, CPU otherwise

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Note

Make sure to configure your audio input device correctly in the `record_audio()` function of `app.py`.
