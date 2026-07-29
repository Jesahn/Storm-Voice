import httpx
import json
import asyncio
from typing import AsyncGenerator, Dict, List, Optional

PERSONALITIES = {
    "investor_pitch": {
        "name": "Investor Pitcher",
        "description": "High-impact, confident AI assistant designed for executive pitch demonstrations.",
        "system_prompt": (
            "You are Storm-Bot, a state-of-the-art intelligent real-time voice assistant developed for the Storm-Voice platform. "
            "Speak clearly, concisely, and with high confidence. Keep responses crisp and punchy (1-3 sentences) suited for spoken voice conversation. "
            "Highlight speed, privacy, and local deployment capabilities when relevant."
        )
    },
    "executive_assistant": {
        "name": "Executive Assistant",
        "description": "Professional, articulate, and highly organized voice assistant.",
        "system_prompt": (
            "You are Storm-Bot, an executive voice assistant. Be ultra-professional, efficient, polite, and helpful. "
            "Keep spoken responses clear and structured."
        )
    },
    "tech_architect": {
        "name": "Tech Architect",
        "description": "Deep technical knowledge with concise architectural explanations.",
        "system_prompt": (
            "You are Storm-Bot, a senior AI architect. Provide technically precise, concise explanations. "
            "Focus on efficiency, zero-latency streaming pipelines, and local AI stack performance."
        )
    },
    "custom": {
        "name": "Custom Storm Persona",
        "description": "Fully customizable Storm-Bot persona.",
        "system_prompt": (
            "You are Storm-Bot, an intelligent real-time voice assistant. Answer naturally and concisely for voice output."
        )
    }
}

class StormLLMClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model_name: str = "gemma-4-e2b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.active_personality = "investor_pitch"

    async def check_health(self) -> Dict:
        """Check connection to LM Studio / OpenAI-compatible local server."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/models")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("id") for m in data.get("data", [])]
                    return {
                        "online": True,
                        "status": "Connected to LM Studio Backend",
                        "available_models": models,
                        "base_url": self.base_url
                    }
        except Exception as e:
            return {
                "online": False,
                "status": f"LM Studio Offline / Unreachable at {self.base_url}",
                "error": str(e)
            }
        return {"online": False, "status": "Invalid Response from LLM Server"}

    async def stream_response(
        self, 
        messages: List[Dict[str, str]], 
        personality_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from LM Studio (Gemma 4 E2B).
        """
        p_key = personality_key or self.active_personality
        sys_prompt = PERSONALITIES.get(p_key, PERSONALITIES["investor_pitch"])["system_prompt"]

        formatted_messages = [{"role": "system", "content": sys_prompt}] + messages

        payload = {
            "messages": formatted_messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 250
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        yield f"Storm System Notice: LLM backend returned status {response.status_code}."
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_obj = json.loads(data_str)
                                delta = data_obj["choices"][0]["delta"]
                                content_chunk = delta.get("content", "")
                                if content_chunk:
                                    yield content_chunk
                            except Exception:
                                continue
        except Exception as err:
            yield f"Storm System Notice: Unable to communicate with local Gemma 4 engine. Ensure LM Studio server is running at {self.base_url}."
