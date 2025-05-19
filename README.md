# 🎙️ WhispererAI 🤖

**An intelligent voice-based AI assistant that transcribes speech and answers questions in real-time using OpenAI's Whisper and Llama models.**

---

<!-- Badges (Placeholder - Generate these from services like shields.io) -->
<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg?style=for-the-badge">
</p>

---

## ✨ Core Features

*   🎤 **Real-time Audio Recording & Transcription**: Capture and convert speech to text instantly.
*   🧠 **Local Speech Recognition**: Utilizes the Whisper Base model for efficient on-device processing.
*   💡 **AI-Powered Responses**: Leverages Llama (via Ollama) for intelligent question answering.
*   🔊 **High-Quality Audio Processing**: Includes noise filtering for clearer audio input.
*   🚀 **CUDA Acceleration**: Supports GPU acceleration for faster performance.
*   💻 **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS.

## 🛠️ Tech Stack

*   **Programming Language**: Python 3.8+
*   **Speech-to-Text**: OpenAI Whisper (Base model)
*   **Language Model**: Llama (via Ollama)
*   **Core Libraries**:
    *   PyTorch
    *   Transformers
    *   SoundFile
*   **Audio Backend**: FFMPEG

## 📋 Prerequisites

*   🐍 Python 3.8 or higher.
*   🎮 CUDA-capable GPU (Optional, but highly recommended for performance).
*   🎞️ FFMPEG installed and accessible in your system's PATH.
*   🦙 Ollama installed and running locally.
*   🎧 A compatible audio input device (Defaults to HyperX Cloud Stinger Core Wireless on Windows, or system default otherwise).

## 🚀 Getting Started: Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/WhispererAI.git
    cd WhispererAI
    ```

2.  **Set Up a Virtual Environment**:
    ```bash
    python -m venv venv
    ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install and Run Ollama**:
    *   Download and install Ollama from [ollama.com](https://ollama.com).
    *   Ensure the Ollama service is running.
    *   Pull the Llama model you intend to use (e.g., `ollama pull llama3.2`).

## 💻 How to Use

1.  **Launch the Application**:
    ```bash
    python app.py
    ```

2.  **Interact with the Assistant**:
    *   Press `R` to **Start Recording** your voice.
    *   Press `S` to **Stop Recording** and process the audio.
    *   Press `C` to **Clear** the terminal screen.
    *   Press `Q` to **Quit** the application.

## ⚙️ Configuration Details

The application comes with the following default settings:

*   **Audio Sample Rate**: 48kHz
*   **Audio Channels**: Mono
*   **Whisper Model**: `openai/whisper-base`
*   **LLM (via Ollama)**: `llama3.2` (Ensure this model is available in your Ollama setup)
*   **Processing Device**: CUDA (if available), otherwise CPU.
*   **Audio Filters**:
    *   High-pass: 50Hz
    *   Low-pass: 15kHz
    *   Volume Boost: 1.5x

## 🎤 Audio Device Setup

*   **Windows**: Attempts to automatically detect "Microphone (HyperX Cloud Stinger Core Wireless DTS)".
*   **Linux/macOS**: Uses the default system audio input device.
*   ℹ️ If the preferred device isn't found, the application will list available audio devices. You may need to modify [`app.py`](app.py:131) to specify your device.

## 🤝 Contributing

Contributions are highly encouraged and welcome! If you have improvements or bug fixes, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

## ⚠️ Important Notes

*   Ensure **Ollama is running** with the specified model before starting WhispererAI.
*   Configure your **audio input device** in [`app.py`](app.py:0) if the default settings don't work for your setup.
*   For the best performance, a **CUDA-capable GPU** is recommended.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details (assuming a LICENSE file exists or will be created).

---

<p align="center">Happy Whispering! 💬</p>
