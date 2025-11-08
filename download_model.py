# download_model.py
from TTS.api import TTS

print("📦 Downloading and initializing Coqui TTS model...")
TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)
print("✅ Download complete.")
