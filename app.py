import os  # Import the os module to read environment variables
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import subprocess
import numpy as np
import soundfile as sf
import termios
import sys
import tty
import openai  # Ensure you have the openai library installed

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create the pipeline for ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to get a single character from stdin without echoing
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Function to record audio
def record_audio():
    # Use ffmpeg to capture audio from the specified output device
    command = [
        'ffmpeg',
        '-f', 'pulse',
        '-i', 'alsa_output.usb-HP__Inc_HyperX_Cloud_Stinger_Core_Wireless_DTS_000000000000-00.analog-stereo.monitor',  # Change this to your specific output device if needed
        '-ar', '16000',  # Set sample rate to 16kHz
        '-ac', '1',      # Set number of audio channels to 1 (mono)
        '-f', 'wav',     # Output format
        'pipe:1'         # Output to stdout
    ]

    # Start the ffmpeg process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    audio_frames = []
    print("Recording...")
    try:
        while True:
            # Read audio data in chunks
            audio_data = process.stdout.read(4096)  # Read in 4096-byte chunks
            if not audio_data:
                break
            
            # Convert byte data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_frames.append(audio_array)

    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        process.terminate()

    # Concatenate audio frames
    if audio_frames:
        audio = np.concatenate(audio_frames, axis=0)
        return audio
    else:
        return np.array([])  # Return an empty array if no audio was recorded

# Function to transcribe audio
def transcribe_audio(audio):
    if audio.size == 0:  # Check if audio is empty
        return "No audio recorded."

    # Save the audio data to a WAV file
    sf.write('recorded_audio.wav', audio, 16000)

    # Read the audio data from the WAV file
    audio_input, _ = sf.read('recorded_audio.wav')

    # Process the audio file for transcription
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)

    # Convert input_features to the expected data type
    input_features = inputs.input_features.to(torch_dtype)

    # Move input_features to the same device as the model
    input_features = input_features.to(device)

    # Generate transcription
    generated_ids = model.generate(input_features=input_features)

    # Decode the generated ids
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]

# Function to check if the transcription is a question
def is_question(transcription):
    return transcription.strip().endswith("?")

# Function to send the transcription to OpenAI and get a response
def ask_openai(question):
    if question.strip():
        print("Sending to OpenAI:", question.strip())
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f'Esti asistent personal si raspunzi corect la toate intrebarile care iti vin: {question.strip()}'}]
        )
        answer = response['choices'][0]['message']['content']
        print("OpenAI's response:")
        print(answer)

if __name__ == "__main__":
    print("Press '+' to start recording, '-' to stop and transcribe, or 'q' to quit.")

    while True:
        char = getch()
        if char == '+':
            audio = record_audio()
            print("Recording completed. Press '-' to transcribe.")
        elif char == '-':
            if 'audio' in locals() and audio is not None:
                transcription = transcribe_audio(audio)
                print(f"Transcription: {transcription}")

                if is_question(transcription):
                    ask_openai(transcription)
                else:
                    print("No question detected. No response generated.")
            else:
                print("No recording found. Press '+' to start recording.")
        elif char == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid key. Press '+' to start recording, '-' to stop and transcribe, or 'q' to quit.")

