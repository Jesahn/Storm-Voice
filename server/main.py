import os
import io
import time
import json
import asyncio
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Force D: drive environment isolation before imports
import server.env_config as env_config
from server.vad import StormVAD
from server.stt import StormSTT
from server.llm_client import StormLLMClient, PERSONALITIES
from server.tts_engine import StormTTSEngine
from server.session_manager import StormSessionManager

app = FastAPI(title="Storm-Voice Core Engine", version="1.0.0")

# Serve static files
STATIC_DIR = env_config.BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Core AI Components
llm_client = StormLLMClient()
stt_engine = StormSTT()
tts_engine = StormTTSEngine()
session_mgr = StormSessionManager()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Storm-Voice Platform Initializing...</h1>")

@app.get("/api/status")
async def get_system_status():
    """Returns Storm-Voice system telemetry for investor UI dashboard."""
    llm_health = await llm_client.check_health()
    voices = tts_engine.list_voice_profiles()
    
    return {
        "system_name": "Storm-Voice",
        "bot_name": "Storm-Bot",
        "isolation_status": "LOCKED_D_DRIVE",
        "base_directory": str(env_config.BASE_DIR),
        "llm_backend": llm_health,
        "stt_engine": {
            "name": "NVIDIA Parakeet / Local STT",
            "model": stt_engine.model_name,
            "status": "READY"
        },
        "tts_engine": {
            "name": "Qwen3-TTS / Neural Voice Synthesizer",
            "active_profile": tts_engine.active_voice_profile,
            "available_profiles_count": len(voices)
        },
        "hardware": {
            "gpu": "NVIDIA GeForce GTX 1060 6GB",
            "ram": "32.0 GB",
            "cpu": "Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz"
        },
        "active_personality": llm_client.active_personality
    }

@app.get("/api/personalities")
async def get_personalities():
    return {
        "active": llm_client.active_personality,
        "personalities": PERSONALITIES
    }

@app.post("/api/personalities/select")
async def select_personality(data: dict):
    p_key = data.get("personality")
    if p_key in PERSONALITIES:
        llm_client.active_personality = p_key
        return {"status": "success", "active": p_key, "info": PERSONALITIES[p_key]}
    raise HTTPException(status_code=400, detail="Invalid personality key")

@app.get("/api/voices")
async def list_voices():
    return {
        "active": tts_engine.active_voice_profile,
        "voices": tts_engine.list_voice_profiles()
    }

@app.get("/api/test-tts")
async def test_tts_endpoint():
    """Generates a test speech audio payload using Kokoro ONNX neural voice."""
    synth_res = tts_engine.synthesize_speech("Hello! I am Storm-Bot. This is a real-time local neural voice test.")
    return synth_res

@app.post("/api/voices/select")
async def select_voice(data: dict):
    v_id = data.get("voice_id")
    tts_engine.active_voice_profile = v_id
    return {"status": "success", "active_voice": v_id}

@app.post("/api/voice-clone")
async def clone_voice(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """Voice Cloning Endpoint: Accepts microphone recording or uploaded audio file."""
    try:
        content = await file.read()
        profile = tts_engine.clone_voice_from_audio(name, content, file.filename)
        return {
            "status": "success",
            "message": f"Voice profile '{name}' successfully cloned and activated!",
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.post("/api/export-logs")
async def export_logs(data: dict):
    fmt = data.get("format", "json")
    result = session_mgr.export_logs(fmt)
    return result

@app.get("/api/logs/download/{filename}")
async def download_log(filename: str):
    log_file = env_config.BASE_DIR / "logs" / filename
    if log_file.exists():
        return FileResponse(log_file, filename=filename)
    raise HTTPException(status_code=404, detail="Log file not found")


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket audio pipeline supporting VAD, streaming STT,
    LM Studio (Gemma 4 E2B) completion streaming, Qwen3-TTS, and Barge-In interruption.
    """
    await websocket.accept()
    print("[Storm WebSocket] Client connected.")

    vad = StormVAD()

    try:
        while True:
            # Receive message (JSON event or Binary Audio Chunk)
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Convert 16kHz int16 PCM bytes to float32 numpy array
                int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
                float32_data = int16_data.astype(np.float32) / 32768.0

                vad_res = vad.process_chunk(float32_data)

                # Send live VAD meter status to browser visualizer
                await websocket.send_json({
                    "type": "vad_meter",
                    "status": vad_res["status"],
                    "rms": vad_res["rms"]
                })

                # Check for Barge-In Interruption if user starts speaking while Storm-Bot is generating/playing
                if vad_res["speech_started"] and session_mgr.is_bot_speaking:
                    session_mgr.trigger_barge_in()
                    await websocket.send_json({"type": "barge_in_stop"})

                # On Speech End (User finished speaking)
                if vad_res["speech_ended"] and vad_res["audio_buffer"] is not None:
                    session_mgr.clear_interrupted()
                    
                    # 1. Transcribe audio via STT
                    user_text = stt_engine.transcribe(vad_res["audio_buffer"])
                    if not user_text.strip():
                        continue

                    session_mgr.add_user_turn(user_text)

                    # Send User Transcript to UI
                    await websocket.send_json({
                        "type": "user_transcript",
                        "text": user_text
                    })

                    # Notify UI that Storm-Bot is thinking
                    await websocket.send_json({"type": "bot_thinking"})

                    # 2. Get history and stream LLM response from Gemma 4 E2B
                    history = session_mgr.get_formatted_history()
                    start_time = time.time()
                    
                    full_bot_text = ""
                    text_sentence_buffer = ""
                    session_mgr.set_bot_speaking(True)

                    async for chunk in llm_client.stream_response(history):
                        if session_mgr.is_interrupted:
                            print("[Storm-Bot] Streaming cancelled due to user interruption.")
                            break

                        full_bot_text += chunk
                        text_sentence_buffer += chunk

                        # Stream partial text to UI
                        await websocket.send_json({
                            "type": "bot_text_chunk",
                            "chunk": chunk
                        })

                        # Synthesize voice for sentence boundaries to achieve low latency
                        if any(punct in text_sentence_buffer for punct in [".", "?", "!", "\n"]):
                            synth_res = tts_engine.synthesize_speech(text_sentence_buffer)
                            text_sentence_buffer = ""

                            if synth_res["audio_base64"]:
                                await websocket.send_json({
                                    "type": "bot_audio_chunk",
                                    "audio": synth_res["audio_base64"],
                                    "duration": synth_res["duration"]
                                })

                    # Synthesize any remaining trailing text buffer
                    if text_sentence_buffer.strip() and not session_mgr.is_interrupted:
                        synth_res = tts_engine.synthesize_speech(text_sentence_buffer)
                        if synth_res["audio_base64"]:
                            await websocket.send_json({
                                "type": "bot_audio_chunk",
                                "audio": synth_res["audio_base64"],
                                "duration": synth_res["duration"]
                            })

                    session_mgr.set_bot_speaking(False)

                    if full_bot_text.strip():
                        latency_ms = (time.time() - start_time) * 1000.0
                        session_mgr.add_bot_turn(
                            full_bot_text, 
                            latency_ms=latency_ms,
                            voice_profile=tts_engine.active_voice_profile
                        )

                    await websocket.send_json({"type": "bot_finished"})

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("action") == "interrupt":
                    session_mgr.trigger_barge_in()
                    await websocket.send_json({"type": "barge_in_stop"})

    except WebSocketDisconnect:
        print("[Storm WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[Storm WebSocket Error] {e}")
