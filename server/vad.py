import numpy as np

class StormVAD:
    """
    Voice Activity Detection (VAD) processor for Storm-Voice.
    Uses adaptive energy thresholding and signal analysis for low-latency turn-taking & barge-in.
    """
    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.025, silence_duration_ms: int = 700):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration_samples = int((silence_duration_ms / 1000.0) * sample_rate)
        
        self.is_speaking = False
        self.silence_samples_count = 0
        self.speech_buffer = []

    def process_chunk(self, pcm_data: np.ndarray) -> dict:
        """
        Process a chunk of float32 PCM audio (-1.0 to 1.0).
        Returns a dict indicating speech status events.
        """
        if len(pcm_data) == 0:
            return {"status": "silent", "speech_started": False, "speech_ended": False}

        rms = float(np.sqrt(np.mean(pcm_data ** 2)))
        speech_detected = rms > self.energy_threshold

        speech_started = False
        speech_ended = False

        if speech_detected:
            self.silence_samples_count = 0
            self.speech_buffer.append(pcm_data)
            if not self.is_speaking:
                self.is_speaking = True
                speech_started = True
        else:
            if self.is_speaking:
                self.speech_buffer.append(pcm_data)
                self.silence_samples_count += len(pcm_data)
                if self.silence_samples_count >= self.silence_duration_samples:
                    self.is_speaking = False
                    speech_ended = True

        full_audio = None
        if speech_ended and self.speech_buffer:
            full_audio = np.concatenate(self.speech_buffer)
            self.speech_buffer = []
            
            # Require at least 0.5s of audio to avoid processing short noise clicks
            if len(full_audio) < int(0.5 * self.sample_rate):
                full_audio = None
                speech_ended = False

        return {
            "status": "speaking" if self.is_speaking else "silent",
            "rms": rms,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "audio_buffer": full_audio
        }

    def reset(self):
        self.is_speaking = False
        self.silence_samples_count = 0
        self.speech_buffer = []
