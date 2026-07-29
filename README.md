# ⚡ Storm-Voice

**Storm-Voice** is a sleek, high-performance real-time desktop & web voice platform powered by **Storm-Bot**, an intelligent AI voice assistant. Built for zero-latency local execution, real-time speech recognition, Edge-TTS neural speech synthesis, and acoustic voice cloning.

---

## 🌟 Key Features

- 🎙️ **Real-Time Speech-to-Text (STT)**: Hugging Face Whisper ASR (`openai/whisper-tiny`) with noise-gating and anti-hallucination repetition filtering.
- 🧠 **LLM Intelligence Backend**: Connects locally to **LM Studio** (`Gemma 4 E2B`) on `http://localhost:1234/v1`.
- 🔊 **Neural Speech Synthesis (TTS)**: Studio-quality Microsoft Edge AI Neural Voices (`Ava`, `Andrew`, `Sonia`) and SAPI5 offline fallback.
- 🎤 **Acoustic Voice Cloning Studio**: Clone any custom voice persona from an uploaded `.wav`/`.mp3` audio sample or live microphone recording.
- ⚡ **Acoustic Echo Cancellation (AEC)**: Hardware echo suppression and dynamic playback muting for seamless full-duplex conversation.
- 📄 **Presentation HUD & Session Export**: Real-time VAD spectrum visualizer, 3D reactor orb animation, and exportable JSON/Markdown transcripts.
- 🔒 **Drive Isolation**: Strict D: Drive cache isolation preventing C: drive storage bloat.

---

## 🚀 Quick Start Guide

### Prerequisites
- Windows 10/11 64-bit
- Python 3.10+
- LM Studio running **Gemma 4 E2B** on `http://localhost:1234/v1`

### Running the App
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Storm-Voice.git
   cd Storm-Voice
   ```
2. Double-click `launch_storm_voice.bat` or run:
   ```bash
   .\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
   ```
3. Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🛠️ Tech Stack

- **Core Server**: FastAPI, Uvicorn, WebSockets
- **STT**: Hugging Face Transformers (`openai/whisper-tiny`), PyTorch
- **TTS & Voice Clone**: Edge-TTS, PyTTSx3, PyDub, SciPy
- **Frontend**: Vanilla HTML5, Modern CSS Glassmorphic UI, WebAudio API
