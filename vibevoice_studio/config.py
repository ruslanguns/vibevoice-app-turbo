"""Configuration constants for VibeVoice Studio."""

from __future__ import annotations

import os

# ── Model IDs ──────────────────────────────────────────────────────────
ASR_MODEL_ID: str = "microsoft/VibeVoice-ASR-HF"
TTS_MODEL_ID: str = "microsoft/VibeVoice-Realtime-0.5B"

# ── Hardware ───────────────────────────────────────────────────────────
DEVICE: str = os.environ.get("VIBEVOICE_DEVICE", "auto")
DTYPE: str = os.environ.get("VIBEVOICE_DTYPE", "float16")

# ── Audio ──────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 24_000
MAX_ASR_DURATION_SECONDS: int = 3600  # 60 minutes

# ── ASR settings ───────────────────────────────────────────────────────
ASR_TOKENIZER_CHUNK_SIZE: int = 64_000  # default for 60s @ 24kHz

# ── TTS settings ───────────────────────────────────────────────────────
TTS_DEFAULT_SPEAKER: str = "Carter"
TTS_SUPPORTED_LANGUAGES: list[str] = [
    "English", "German", "French", "Italian", "Japanese",
    "Korean", "Dutch", "Polish", "Portuguese", "Spanish",
]

# ── Server ─────────────────────────────────────────────────────────────
GRADIO_SERVER_NAME: str = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE: bool = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
