import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from server.env_config import BASE_DIR, TMP_DIR

class StormSessionManager:
    """
    Manages active conversation session, barge-in cancellation signals,
    telemetry metrics, and exportable conversation logs for Storm-Voice.
    """
    def __init__(self):
        self.session_id = f"storm_session_{int(time.time())}"
        self.start_time = time.time()
        self.history: List[Dict] = []
        self.is_interrupted = False
        self.is_bot_speaking = False

    def trigger_barge_in(self):
        """Sets interruption flag to cancel ongoing LLM streaming / TTS playback."""
        print("[Storm Session] Barge-in triggered! Halting playback & stream generation.")
        self.is_interrupted = True
        self.is_bot_speaking = False

    def set_bot_speaking(self, status: bool):
        self.is_bot_speaking = status

    def clear_interrupted(self):
        self.is_interrupted = False

    def add_user_turn(self, text: str):
        if not text.strip():
            return
        turn = {
            "speaker": "User",
            "text": text.strip(),
            "timestamp": time.strftime("%H:%M:%S"),
            "unix_time": time.time()
        }
        self.history.append(turn)
        return turn

    def add_bot_turn(self, text: str, latency_ms: float = 0.0, voice_profile: str = "default"):
        if not text.strip():
            return
        turn = {
            "speaker": "Storm-Bot",
            "text": text.strip(),
            "timestamp": time.strftime("%H:%M:%S"),
            "unix_time": time.time(),
            "latency_ms": round(latency_ms, 2),
            "voice_profile": voice_profile
        }
        self.history.append(turn)
        return turn

    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Formats conversation turns into OpenAI API message list format."""
        formatted = []
        for item in self.history:
            role = "user" if item["speaker"] == "User" else "assistant"
            formatted.append({"role": role, "content": item["text"]})
        return formatted

    def export_logs(self, format_type: str = "json") -> Dict:
        """Exports full conversation logs to JSON or Markdown on D: drive."""
        export_dir = BASE_DIR / "logs"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.session_id}.{format_type}"
        file_path = export_dir / filename

        session_meta = {
            "session_id": self.session_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "duration_seconds": round(time.time() - self.start_time, 2),
            "total_turns": len(self.history),
            "system": "Storm-Voice Real-Time AI System"
        }

        if format_type == "json":
            export_content = {
                "metadata": session_meta,
                "transcript": self.history
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_content, f, indent=2)
        else: # markdown
            md_lines = [
                f"# Storm-Voice Conversation Log - {self.session_id}",
                f"**Date**: {session_meta['start_time']}",
                f"**Duration**: {session_meta['duration_seconds']} seconds",
                f"**Total Utterances**: {session_meta['total_turns']}",
                "\n---",
                "## Transcript\n"
            ]
            for turn in self.history:
                spk = turn["speaker"]
                txt = turn["text"]
                ts = turn["timestamp"]
                md_lines.append(f"**[{ts}] {spk}**: {txt}\n")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))

        return {
            "file_name": filename,
            "file_path": str(file_path),
            "download_url": f"/api/logs/download/{filename}",
            "metadata": session_meta
        }
