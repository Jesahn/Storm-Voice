import os
import sys
import numpy as np
from transformers import pipeline

from server.env_config import BASE_DIR, TMP_DIR

class StormSTT:
    """
    Real Speech Recognition Engine for Storm-Voice using Hugging Face Whisper ASR.
    Transcribes microphone audio streams in real-time with anti-hallucination filtering.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        self.model_name = model_name
        self.asr_pipeline = None

    def _initialize_engine(self):
        try:
            print(f"[Storm-STT] Loading real local Whisper STT model ({self.model_name})...")
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device="cpu"
            )
            print("[Storm-STT] STT Engine ready!")
        except Exception as e:
            print(f"[Storm-STT Error] Failed to load ASR pipeline: {e}")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribes float32 PCM numpy audio array to real text.
        """
        if len(audio_data) == 0:
            return ""

        # Noise / Silence RMS Gate (0.008 allows normal mic speech while ignoring total silence)
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        if rms < 0.008:
            print(f"[Storm-STT] Ignored low energy chunk (RMS: {rms:.4f})")
            return ""

        try:
            if self.asr_pipeline is None:
                self._initialize_engine()

            if self.asr_pipeline:
                audio_float = audio_data.astype(np.float32)
                
                result = self.asr_pipeline(
                    {"raw": audio_float, "sampling_rate": sample_rate},
                    generate_kwargs={"language": "english"}
                )
                text = result.get("text", "").strip()

                # Anti-Hallucination Filter: Drop Whisper subtitle/noise artifacts
                text_clean = text.lower().strip("!.,? ")
                
                # Check for pure dots / punctuation
                if not any(c.isalnum() for c in text):
                    print(f"[Storm-STT] Filtered punctuation-only noise: '{text}'")
                    return ""

                # Known Whisper-tiny noise hallucinations
                hallucinations = {
                    "thank you for watching", "thanks for watching", "thank you for watching!",
                    "thanks for watching!", "thank you very much", "thank you", "thanks",
                    "subtitles by", "amara.org", "subscribe", "like and subscribe",
                    "bye", "goodbye", "thank you.", "thanks."
                }
                if text_clean in hallucinations or any(h in text_clean for h in ["thank you for watching", "thanks for watching", "subtitles by"]):
                    print(f"[Storm-STT] Filtered Whisper hallucination phrase: '{text}'")
                    return ""

                # Anti-Hallucination Filter: Detect repetitive token loops (e.g. "hey, hey, hey...")
                words = text.split()
                if len(words) > 10:
                    first_word = words[0].lower().strip(",.!?")
                    repetition_count = sum(1 for w in words if w.lower().strip(",.!?") == first_word)
                    if repetition_count / len(words) > 0.6:
                        print(f"[Storm-STT] Filtered Whisper repetition hallucination: '{text[:40]}...'")
                        return ""

                print(f"[Storm-STT Output] Transcribed: '{text}'")
                return text
        except Exception as err:
            print(f"[Storm STT Error] Transcription exception: {err}")
        
        return ""
