"""VibeVoice-ASR wrapper for speaker-diariazed transcription.

Uses the HuggingFace Transformers-compatible model (VibeVoice-ASR-HF)
available in transformers >= 5.3.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import (
    ASR_MODEL_ID,
    ASR_TOKENIZER_CHUNK_SIZE,
    DEVICE,
    DTYPE,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single segment of the transcription output."""

    start: float
    end: float
    speaker: int
    content: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "Start": self.start,
            "End": self.end,
            "Speaker": self.speaker,
            "Content": self.content,
        }


@dataclass
class TranscriptionResult:
    """Full transcription result with metadata."""

    segments: list[TranscriptionSegment]
    raw_output: str
    text_only: str

    def to_json(self) -> list[dict[str, float | int | str]]:
        return [seg.to_dict() for seg in self.segments]

    def to_text(self) -> str:
        return self.text_only


class ASREngine:
    """Wrapper around VibeVoice-ASR via HuggingFace Transformers.

    Loads the model lazily on first use to keep import-time fast.
    """

    def __init__(
        self,
        model_id: str = ASR_MODEL_ID,
        device: str = DEVICE,
        dtype: str = DTYPE,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        """Lazily load the ASR model and processor."""
        if self._model is not None:
            return

        logger.info("Loading ASR model: %s", self.model_id)

        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=self.dtype,  # type: ignore[arg-type]
        )

        logger.info("ASR model loaded on %s", self._model.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio: str,
        hotwords: Optional[list[str]] = None,
        chunk_size: int = ASR_TOKENIZER_CHUNK_SIZE,
    ) -> TranscriptionResult:
        """Transcribe an audio file or URL.

        Args:
            audio: Path to audio file or URL.
            hotwords: Optional list of domain-specific hotwords for better accuracy.
            chunk_size: Tokenizer chunk size (default 64000 ≈ 60s @ 24kHz).

        Returns:
            TranscriptionResult with segments, raw output, and text-only string.
        """
        self._load_model()
        assert self._processor is not None
        assert self._model is not None

        kwargs: dict = {
            "audio": audio,
        }
        if hotwords:
            kwargs["hotwords"] = hotwords
        if chunk_size != ASR_TOKENIZER_CHUNK_SIZE:
            kwargs["tokenizer_chunk_size"] = chunk_size

        inputs = self._processor.apply_transcription_request(**kwargs)
        inputs = inputs.to(self._model.device, self._model.dtype)

        output_ids = self._model.generate(**inputs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Get raw output
        raw_output = self._processor.decode(generated_ids)[0]

        # Try parsed format
        try:
            parsed = self._processor.decode(generated_ids, return_format="parsed")[0]
            segments = [
                TranscriptionSegment(
                    start=float(seg["Start"]),
                    end=float(seg["End"]),
                    speaker=int(seg["Speaker"]),
                    content=str(seg["Content"]),
                )
                for seg in parsed
            ]
        except Exception:
            logger.warning("Parsed format failed, falling back to raw output")
            segments = [
                TranscriptionSegment(
                    start=0.0, end=0.0, speaker=0, content=raw_output
                )
            ]

        # Get text-only
        try:
            text_only = self._processor.decode(generated_ids, return_format="transcription_only")[0]
        except Exception:
            text_only = raw_output

        return TranscriptionResult(
            segments=segments,
            raw_output=raw_output,
            text_only=text_only,
        )
