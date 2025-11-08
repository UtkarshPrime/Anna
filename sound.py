import traceback
import re
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample

# ---- Initialize Coqui TTS ----
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", gpu=True)

# ---- Initialize SpeechRecognition ----
try:
    import speech_recognition as sr
except ImportError:
    raise ImportError("❌ SpeechRecognition not installed. Install with: pip install SpeechRecognition")

# ---- Voice Input Function ----
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
    except Exception as e:
        print("❌ Microphone not accessible:", e)
        return None

    try:
        text = recognizer.recognize_google(audio)
        print("📝 You said:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand your speech.")
    except sr.RequestError:
        print("❌ Could not request results. Check your internet.")
    return None

# ---- Speak Function with Emotion Carrying Only for First Sentence ----
def clean_and_group_by_emotion(text: str):
    parts = re.split(r'(\*(?:whisper|sad|quiet|soft|excited|happy|fast|loud)\*)', text, flags=re.IGNORECASE)
    grouped_blocks = []
    current_emotion = "neutral"
    buffer = ""

    for part in parts:
        if re.match(r'\*(.*?)\*', part):  # Emotion marker
            if buffer:
                grouped_blocks.append((current_emotion, buffer.strip()))
                buffer = ""
            current_emotion = re.sub(r'[\*\s]', '', part).lower()
        else:
            buffer += " " + part

    if buffer:
        grouped_blocks.append((current_emotion, buffer.strip()))

    return grouped_blocks

def emphasize_text(text: str):
    words = text.split()
    emphasized = []
    for word in words:
        if word.endswith("..."):
            print(f"💬 Emphasizing word: {word.rstrip('.')}")
            emphasized.append(word.rstrip('.'))  # Remove dots, speak slower
        else:
            emphasized.append(word)
    return " ".join(emphasized)

def speak_text(text: str):
    if not text:
        print("⚠️ No text to speak.")
        return

    try:
        # 1. Group text by emotion blocks
        blocks = clean_and_group_by_emotion(text)

        for emotion, content in blocks:
            content = re.sub(r"\*(.*?)\*", "", content).strip()
            content = emphasize_text(content)

            # Set speed based on emotion
            if emotion in ["whisper", "sad", "quiet", "soft"]:
                speed = 1.0
            elif emotion in ["excited", "happy", "fast", "loud"]:
                speed = 1.5
            else:
                speed = 1.25

            if content:
                print(f"🗣 Speaking ({emotion}):", content)
                tts.tts_to_file(
                    text=content,
                    file_path="output.wav",
                    speed=speed,
                    emotion="excited"
                )

                sd.default.samplerate = 48000
                # Read TTS output
                data, samplerate = sf.read("output.wav", dtype='float32')
                
                # Target sample rate (Hi-Fi Cable default)
                target_sr = int(sd.query_devices(46)['default_samplerate'])

                # Resample if needed
                if samplerate != target_sr:
                    import numpy as np
                    num_samples = round(len(data) * float(target_sr) / samplerate)
                    data = resample(data, num_samples)
                    samplerate = target_sr

                sd.play(data, samplerate, device=37, blocking=True)
                sd.wait()

    except Exception as e:
        
        print("❌ Error during TTS or playback:", e)
        traceback.print_exc()

# ---- Stop Function ----
def stop_speaking():
    # Coqui TTS doesn't stream audio live, no stop functionality needed
    pass

# ---- Test Locally ----
if __name__ == "__main__":
    result = listen_and_transcribe()
    if result:
        speak_text(result)
