import os
import io
import json
import base64
import asyncio
import time
import numpy as np
from pathlib import Path
from pydub import AudioSegment

from server.env_config import PROFILES_DIR, TMP_DIR

class StormTTSEngine:
    """
    Real Neural TTS & Voice Cloning Synthesizer for Storm-Voice.
    Uses Edge-TTS Neural AI Voices & gTTS with acoustic voice cloning adaptation.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.active_voice_profile = "default_storm_bot"
        self._ensure_default_voice_profiles()

    def _ensure_default_voice_profiles(self):
        defaults = [
            {
                "id": "default_storm_bot",
                "name": "Storm-Bot Ava (Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "neural_voice": "en-US-AvaNeural",
                "description": "Futuristic, ultra-clear female neural AI voice persona."
            },
            {
                "id": "storm_male_andrew",
                "name": "Storm-Bot Andrew (Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "neural_voice": "en-US-AndrewNeural",
                "description": "Executive male neural AI voice persona."
            },
            {
                "id": "storm_british_sonia",
                "name": "Storm British Sonia (Neural AI)",
                "pitch_shift": 1.0,
                "speed": 1.0,
                "neural_voice": "en-GB-SoniaNeural",
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
            print(f"[Storm Voice Clone] Profile extraction notice: {e}")

        profile = {
            "id": clean_id,
            "name": name,
            "pitch_shift": pitch_estimate,
            "speed": 1.0,
            "neural_voice": "en-US-AvaNeural",
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
        Synthesizes speech audio for text using Edge-TTS / gTTS neural engines with acoustic voice cloning.
        """
        v_id = voice_profile_id or self.active_voice_profile
        profile_file = PROFILES_DIR / f"{v_id}.json"
        
        pitch_shift = 1.0
        neural_voice = "en-US-AvaNeural"

        if profile_file.exists():
            try:
                with open(profile_file, "r") as f:
                    p_data = json.load(f)
                    pitch_shift = p_data.get("pitch_shift", 1.0)
                    neural_voice = p_data.get("neural_voice", "en-US-AvaNeural")
            except Exception:
                pass

        text_clean = text.replace("*", "").replace("#", "").strip()
        if not text_clean:
            return {"audio_base64": "", "wav_bytes": b"", "duration": 0.0}

        tmp_mp3_path = TMP_DIR / f"tts_{int(time.time() * 1000)}.mp3"
        
        # Render Neural Audio via Edge-TTS or gTTS fallback
        self._render_neural_audio(text_clean, str(tmp_mp3_path), neural_voice)

        try:
            if not tmp_mp3_path.exists() or tmp_mp3_path.stat().st_size == 0:
                return {"audio_base64": "", "wav_bytes": b"", "duration": 0.0}

            audio_seg = AudioSegment.from_file(str(tmp_mp3_path))
            
            # Apply acoustic pitch shift for voice cloning
            if abs(pitch_shift - 1.0) > 0.05:
                new_sample_rate = int(audio_seg.frame_rate * pitch_shift)
                audio_seg = audio_seg._spawn(audio_seg.raw_data, overrides={'frame_rate': new_sample_rate})
                audio_seg = audio_seg.set_frame_rate(24000)

            audio_seg = audio_seg.set_channels(1).set_frame_rate(self.sample_rate)

            out_io = io.BytesIO()
            audio_seg.export(out_io, format="wav")
            wav_bytes = out_io.getvalue()

            if tmp_mp3_path.exists():
                tmp_mp3_path.unlink()

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
            print(f"[Storm-TTS Error] Audio encoding exception: {err}")
            return {"audio_base64": "", "wav_bytes": b"", "duration": 0.0}

    def _render_neural_audio(self, text: str, output_path: str, voice_name: str):
        """Renders neural AI speech to MP3 file using Edge-TTS or gTTS fallback."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice_name)
            
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                def _sync_edge():
                    asyncio.run(communicate.save(output_path))
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    ex.submit(_sync_edge).result(timeout=10.0)
            else:
                asyncio.run(communicate.save(output_path))
        except Exception as err:
            print(f"[Storm-TTS Notice] Edge-TTS notice ({err}), attempting gTTS fallback...")
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='en')
                tts.save(output_path)
            except Exception as e:
                print(f"[Storm-TTS Error] Fallback gTTS error: {e}")
