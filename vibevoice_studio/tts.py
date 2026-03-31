"""VibeVoice-Realtime-TTS wrapper for real-time speech synthesis.

Wraps the VibeVoice-Realtime-0.5B model from the microsoft/VibeVoice
repository for streaming text-to-speech generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import torchaudio

from .config import (
    DEVICE,
    DTYPE,
    SAMPLE_RATE,
    TTS_DEFAULT_SPEAKER,
    TTS_SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)

# Available speakers bundled with the model
DEFAULT_SPEAKERS: list[str] = ["Carter"]


@dataclass
class TTSSpeaker:
    """Information about a TTS speaker/voice."""

    name: str
    language: str
    is_experimental: bool = False


@dataclass
class TTSResult:
    """Result from TTS synthesis."""

    audio: np.ndarray
    sample_rate: int
    speaker: str

    def save(self, path: str | Path) -> None:
        """Save audio to a WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(self.audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(str(path), tensor, self.sample_rate)
        logger.info("Audio saved to %s", path)


class TTSEngine:
    """Wrapper around VibeVoice-Realtime-0.5B.

    Uses the model from the microsoft/VibeVoice repository.
    Supports streaming text input for real-time synthesis.
    """

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: str = DEVICE,
        dtype: str = DTYPE,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazily load the TTS model."""
        if self._model_loaded:
            return

        logger.info("Loading TTS model: %s", self.model_path)

        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            ).to(self.device)
            logger.info("TTS model loaded via transformers")
        except Exception as e:
            logger.error("Failed to load TTS model: %s", e)
            raise RuntimeError(
                f"Could not load TTS model '{self.model_path}'. "
                f"Ensure you have a GPU with sufficient VRAM and the model is available. "
                f"Original error: {e}"
            ) from e

        self._model_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def synthesize(
        self,
        text: str,
        speaker: str = TTS_DEFAULT_SPEAKER,
    ) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            speaker: Speaker name to use.

        Returns:
            TTSResult with audio array and metadata.
        """
        self._load_model()
        assert self._model is not None

        audio = self._model.synthesize(text, speaker=speaker)
        return TTSResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            speaker=speaker,
        )

    def synthesize_stream(
        self,
        text: str,
        speaker: str = TTS_DEFAULT_SPEAKER,
        chunk_size: int = 100,
    ) -> Generator[np.ndarray, None, None]:
        """Stream-synthesize speech from text.

        Yields audio chunks as they're generated, enabling real-time playback.

        Args:
            text: Input text to synthesize.
            speaker: Speaker name to use.
            chunk_size: Number of characters to process per chunk.

        Yields:
            numpy arrays of audio samples.
        """
        self._load_model()
        assert self._model is not None

        # Split text into chunks for streaming
        words = text.split()
        buffer = ""
        for word in words:
            buffer = f"{buffer} {word}" if buffer else word
            if len(buffer) >= chunk_size:
                audio_chunk = self._model.synthesize_chunk(buffer, speaker=speaker)
                yield audio_chunk
                buffer = ""

        # Flush remaining buffer
        if buffer.strip():
            audio_chunk = self._model.synthesize_chunk(buffer, speaker=speaker)
            yield audio_chunk

    @staticmethod
    def get_available_speakers() -> list[TTSSpeaker]:
        """Get list of available speakers.

        Returns default speaker plus experimental multilingual speakers
        if the voices have been downloaded.
        """
        speakers = [TTSSpeaker(name="Carter", language="English")]
        voices_dir = Path("speakers")
        if voices_dir.exists():
            for voice_file in sorted(voices_dir.glob("*.pt")):
                name = voice_file.stem
                lang = "English"
                is_experimental = False
                for supported_lang in TTS_SUPPORTED_LANGUAGES:
                    if supported_lang.lower() in name.lower():
                        lang = supported_lang
                        is_experimental = True
                        break
                speakers.append(TTSSpeaker(name=name, language=lang, is_experimental=is_experimental))
        return speakers
