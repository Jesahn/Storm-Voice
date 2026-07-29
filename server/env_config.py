import os
import sys
from pathlib import Path

# Base project directory on D: drive
BASE_DIR = Path(__file__).resolve().parent.parent

# Cache and tmp paths
CACHE_DIR = BASE_DIR / ".cache"
TMP_DIR = BASE_DIR / "tmp"
HF_CACHE = CACHE_DIR / "huggingface"
TORCH_CACHE = CACHE_DIR / "torch"
PIP_CACHE = CACHE_DIR / "pip"
PROFILES_DIR = BASE_DIR / "voice_profiles"
CONFIG_DIR = BASE_DIR / "config"

for folder in [CACHE_DIR, TMP_DIR, HF_CACHE, TORCH_CACHE, PIP_CACHE, PROFILES_DIR, CONFIG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Enforce environment variables to isolate C: drive completely
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)
os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

print(f"[Storm-Voice Environment] Base Directory: {BASE_DIR}")
print(f"[Storm-Voice Environment] Isolation Locked to D: Drive Cache ({CACHE_DIR})")
