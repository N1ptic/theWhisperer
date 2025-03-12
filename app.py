import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import subprocess
import numpy as np
import soundfile as sf
import sys
from ollama import Client
import platform
import time
from datetime import datetime

if platform.system() == 'Windows':
    import msvcrt
else:
    import termios
    import tty

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header():
    """Print the application header."""
    clear_screen()
    print("=" * 60)
    print("🎤 Voice Assistant powered by Whisper and Llama2".center(60))
    print("=" * 60)
    print("\nSystem Status:")
    print(f"🖥️  Using device: {device}")
    print(f"🤖 LLM Model: {MODEL_NAME}")
    print(f"🎯 Whisper Model: {model_id}")
    print("\n" + "=" * 60 + "\n")

def print_commands():
    """Print available commands."""
    print("\nAvailable Commands:")
    print("  [R] - Start Recording")
    print("  [S] - Stop Recording")
    print("  [C] - Clear screen")
    print("  [Q] - Quit application")
    print("\nWaiting for command...")

# Initialize Ollama client
try:
    client = Client(host='http://localhost:11434')
    MODEL_NAME = "llama3.2"
except Exception as e:
    print("❌ Error: Could not connect to Ollama. Make sure it's running.")
    print(f"Error details: {e}")
    sys.exit(1)

# Set device and dtype for Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor
print("🔄 Loading Whisper model...")
model_id = "openai/whisper-base"  # Using smaller, faster model
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
except Exception as e:
    print("❌ Error: Could not load Whisper model.")
    print(f"Error details: {e}")
    sys.exit(1)

# Create the pipeline for ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs={"use_cache": True},
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
        "max_new_tokens": 128,
        "num_beams": 1,
        "return_timestamps": False,
        "forced_decoder_ids": None  # Remove conflicting forced_decoder_ids
    }
)

def getch():
    """Get a single character from the user."""
    if platform.system() == 'Windows':
        return msvcrt.getch().decode('utf-8')
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def list_audio_devices():
    """List all available audio input devices."""
    try:
        if platform.system() == 'Windows':
            # Use ffmpeg to list devices
            result = subprocess.run([
                'ffmpeg', '-list_devices', 'true',
                '-f', 'dshow', '-i', 'dummy'
            ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            print("\n🎤 Available Audio Devices:")
            devices = result.stderr.decode('utf-8', errors='ignore')
            print(devices)
            return devices
    except Exception as e:
        print(f"❌ Failed to list audio devices: {e}")
        return None

def record_audio(filename="temp.wav"):
    """Start recording audio from HyperX microphone."""
    try:
        print("\n🎙️ Recording from HyperX microphone... Press 'S' to stop")
        
        if platform.system() == 'Windows':
            device = 'audio=Microphone (HyperX Cloud Stinger Core Wireless DTS)'
            command = [
                'ffmpeg',
                '-f', 'dshow',
                '-i', device,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '48000',  # Higher sample rate for better quality
                '-af', 'volume=1.5,highpass=f=50,lowpass=f=15000',  # Audio filters for clearer voice
                '-y',
                filename
            ]
            
            print(f"\n🔄 Using device: {device}")
            
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            time.sleep(0.5)
            if process.poll() is None:
                print(f"✅ Successfully connected to {device}")
                return process
            else:
                print("\n❌ Failed to connect to HyperX microphone.")
                print("\nAvailable devices are:")
                list_audio_devices()
                return None
                
        else:
            # Linux/macOS implementation remains unchanged
            command = [
                'ffmpeg',
                '-f', 'pulse',
                '-i', 'default',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '48000',
                '-af', 'volume=1.5,highpass=f=50,lowpass=f=15000',
                '-y',
                filename
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return process
            
    except Exception as e:
        print(f"\n❌ Recording failed: {e}")
        return None

def stop_recording(process):
    """Stop the recording process."""
    if platform.system() == 'Windows':
        try:
            # Kill all ffmpeg processes forcefully
            subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         errors='ignore')
            return True
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    else:
        try:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            return True
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper."""
    try:
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
            
        # Load and normalize audio
        audio, sr = sf.read(audio_file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Pass the correct format to the pipeline
        result = pipe(
            {
                "raw": audio,
                "sampling_rate": sr
            },
            batch_size=8,
            return_timestamps=False,
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "max_new_tokens": 128,
            }
        )
        
        return result["text"].strip()
            
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        return None

def get_llm_response(prompt):
    """Get response from local Llama model using Ollama."""
    try:
        if not prompt:
            print("❌ Empty prompt received")
            return "Error: No input text to process"
            
        print("🤖 Thinking...")
        response = client.chat(model=MODEL_NAME, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        return response['message']['content']
    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        return f"Error communicating with Ollama: {e}"

def save_conversation(transcription, response):
    """Save conversation to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = "conversation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    with open(f"{log_dir}/conversations.txt", "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n")
        f.write(f"User: {transcription}\n")
        f.write(f"Assistant: {response}\n")
        f.write("-" * 50 + "\n")

def main():
    recording_process = None
    audio_file = "temp.wav"
    
    # Ensure no ffmpeg processes are running at start
    if platform.system() == 'Windows':
        subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL,
                     errors='ignore')
    
    print_header()
    print_commands()
    
    while True:
        key = getch().lower()
        
        if key == 'q':
            if recording_process:
                stop_recording(recording_process)
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass
            print("\n👋 Goodbye!")
            break
            
        elif key == 'r' and not recording_process:
            # Clean up any existing audio file
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass
            recording_process = record_audio(audio_file)
                
        elif key == 's' and recording_process:
            print("\n🛑 Stopping recording...")
            success = stop_recording(recording_process)
            recording_process = None
            
            if success:
                # Wait a bit longer for file to be written
                time.sleep(1.5)
                
                if os.path.exists(audio_file):
                    print("\n🔄 Processing audio file...")
                    transcription = transcribe_audio(audio_file)
                    
                    if transcription:
                        print(f"\n📝 Transcription: {transcription}")
                        
                        print("\n🤖 Getting LLM response...")
                        response = get_llm_response(transcription)
                        print(f"\n💬 Response: {response}")
                        
                        save_conversation(transcription, response)
                    else:
                        print("❌ Transcription failed or returned empty")
                else:
                    print(f"❌ Audio file not found: {audio_file}")
                
                # Clean up
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except Exception as e:
                    print(f"❌ Failed to clean up audio file: {e}")
                
                print("\n" + "=" * 60)
                print_commands()
            else:
                print("\n❌ Failed to stop recording properly")
                print_commands()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        print("\nThank you for using Voice Assistant!")  # Call the main function when the script is run directly

