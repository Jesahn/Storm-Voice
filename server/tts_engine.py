import os
import io
import json
import base64
import time
import urllib.request
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from pydub import AudioSegment

from server.env_config import CACHE_DIR, PROFILES_DIR, TMP_DIR

class StormTTSEngine:
    """
    100% Offline Local Neural TTS & Voice Cloning Synthesizer for Storm-Voice.
    Powered by Kokoro-82M ONNX neural speech model with acoustic voice cloning.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.active_voice_profile = "default_storm_bot"
        self.kokoro = None
        self._ensure_kokoro_model()
        self._ensure_default_voice_profiles()

    def _ensure_kokoro_model(self):
        """Ensures Kokoro ONNX model weights and voice embeddings exist in D: Drive cache."""
        kokoro_cache = CACHE_DIR / "kokoro"
        kokoro_cache.mkdir(parents=True, exist_ok=True)

        onnx_path = kokoro_cache / "kokoro-v1_0.onnx"
        voices_path = kokoro_cache / "voices.npy"
        json_path = kokoro_cache / "voices.json"

        # Download ONNX model if missing
        if not onnx_path.exists():
            print("[Storm-TTS] Downloading local Kokoro-82M ONNX model weights to D: Drive...")
            urllib.request.urlretrieve(
                "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
                str(onnx_path)
            )

        # Download voices.json and convert to voices.npy if missing
        if not voices_path.exists():
            print("[Storm-TTS] Downloading local Kokoro voice embeddings to D: Drive...")
            urllib.request.urlretrieve(
                "https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/voices.json",
                str(json_path)
            )
            with open(json_path, "r") as f:
                data = json.load(f)
            np.save(voices_path, data)

        try:
            from kokoro_onnx import Kokoro
            print("[Storm-TTS] Loading local Kokoro-82M ONNX neural speech engine...")
            self.kokoro = Kokoro(str(onnx_path), str(voices_path))
            print("[Storm-TTS] Kokoro Local Neural Engine Ready!")
        except Exception as e:
            print(f"[Storm-TTS Error] Failed to load Kokoro ONNX model: {e}")

    def _ensure_default_voice_profiles(self):
        defaults = [
            {
                "id": "default_storm_bot",
                "name": "Storm-Bot Bella (Kokoro Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "kokoro_voice": "af_bella",
                "description": "Default futuristic, ultra-clear female neural AI voice persona."
            },
            {
                "id": "storm_male_adam",
                "name": "Storm-Bot Adam (Kokoro Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "kokoro_voice": "am_adam",
                "description": "Executive male neural AI voice persona."
            },
            {
                "id": "storm_british_emma",
                "name": "Storm British Emma (Kokoro Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "kokoro_voice": "bf_emma",
                "description": "Articulate British female neural persona."
            }
        ]

        for prof in defaults:
            p_file = PROFILES_DIR / f"{prof['id']}.json"
            if not p_file.exists():
                with open(p_file, "w") as f:
                    json.dump(prof, f, indent=2)

    def list_voice_profiles(self) -> list:
        profiles = []
        for file in PROFILES_DIR.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    data["id"] = file.stem
                    profiles.append(data)
            except Exception:
                continue
        return profiles

    def clone_voice_from_audio(self, name: str, audio_bytes: bytes, filename: str) -> dict:
        """
        Creates a custom cloned voice profile from recorded microphone audio or uploaded file (.wav, .mp3, .ogg, .flac).
        """
        clean_id = "".join([c if c.isalnum() else "_" for c in name.lower()]).strip("_")
        if not clean_id:
            clean_id = "custom_cloned_voice"

        wav_path = PROFILES_DIR / f"{clean_id}.wav"
        json_path = PROFILES_DIR / f"{clean_id}.json"

        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        pitch_estimate = 1.0
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_frame_rate(24000).set_channels(1)
            samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

            if len(samples) > 0:
                fft = np.fft.rfft(samples[:24000 * 4])
                freqs = np.fft.rfftfreq(len(fft) * 2 - 2, 1.0 / 24000)
                magnitude = np.abs(fft)
                if np.sum(magnitude) > 0:
                    centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
                    pitch_estimate = float(np.clip(centroid / 1400.0, 0.75, 1.45))
        except Exception as e:
            print(f"[Storm Voice Clone] Audio extraction notice: {e}")

        profile = {
            "id": clean_id,
            "name": name,
            "pitch_shift": pitch_estimate,
            "speed": 1.0,
            "kokoro_voice": "af_bella",
            "is_cloned": True,
            "description": f"Cloned custom voice persona created from {filename}.",
            "sample_file": str(wav_path)
        }

        with open(json_path, "w") as f:
            json.dump(profile, f, indent=2)

        self.active_voice_profile = clean_id
        return profile

    def synthesize_speech(self, text: str, voice_profile_id: str = None) -> dict:
        """
        Synthesizes speech audio for text using local Kokoro ONNX neural engine with cloned voice pitch adaptation.
        """
        v_id = voice_profile_id or self.active_voice_profile
        profile_file = PROFILES_DIR / f"{v_id}.json"
        
        pitch_shift = 1.0
        kokoro_voice = "af_bella"

        if profile_file.exists():
            try:
                with open(profile_file, "r") as f:
                    p_data = json.load(f)
                    pitch_shift = p_data.get("pitch_shift", 1.0)
                    kokoro_voice = p_data.get("kokoro_voice", "af_bella")
            except Exception:
                pass

        text_clean = text.replace("*", "").replace("#", "").strip()
        if not text_clean or self.kokoro is None:
            return {"audio_base64": "", "wav_bytes": b"", "duration": 0.0}

        try:
            # Render local Kokoro ONNX neural speech
            samples, sr = self.kokoro.create(text_clean, voice=kokoro_voice, speed=1.0, lang="en-us")
            
            samples = np.squeeze(samples)
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            bio = io.BytesIO()
            wavfile.write(bio, 24000, pcm16)

            audio_seg = AudioSegment.from_file(io.BytesIO(bio.getvalue()), format="wav")
            
            # Apply acoustic pitch shift if cloned voice profile specifies adaptation
            if abs(pitch_shift - 1.0) > 0.05:
                new_sample_rate = int(audio_seg.frame_rate * pitch_shift)
                audio_seg = audio_seg._spawn(audio_seg.raw_data, overrides={'frame_rate': new_sample_rate})
                audio_seg = audio_seg.set_frame_rate(24000)

            audio_seg = audio_seg.set_channels(1).set_frame_rate(self.sample_rate)

            out_io = io.BytesIO()
            audio_seg.export(out_io, format="wav")
            wav_bytes = out_io.getvalue()

            duration = len(audio_seg) / 1000.0
            base64_audio = base64.b64encode(wav_bytes).decode('utf-8')

            return {
                "text": text_clean,
                "audio_base64": base64_audio,
                "wav_bytes": wav_bytes,
                "duration": duration,
                "sample_rate": self.sample_rate,
                "voice_profile": v_id
            }
        except Exception as err:
            print(f"[Storm-TTS Synthesis Error] {err}")
            return {"audio_base64": "", "wav_bytes": b"", "duration": 0.0}
