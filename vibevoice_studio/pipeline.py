"""Combined ASR → process → TTS pipeline.

Provides high-level functions for common workflows like:
- Transcribe audio and summarize
- Transcribe audio and re-synthesize with a different voice
- Chain ASR output into TTS input
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .asr import ASREngine, TranscriptionResult
from .tts import TTSEngine, TTSResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a full pipeline execution."""

    transcription: TranscriptionResult
    synthesis: Optional[TTSResult] = None
    summary: Optional[str] = None


def transcribe_and_synth(
    audio_path: str,
    asr_engine: ASREngine,
    tts_engine: TTSEngine,
    speaker: str = "Carter",
    hotwords: Optional[list[str]] = None,
) -> PipelineResult:
    """Transcribe audio and synthesize the result as speech.

    Useful for voice conversion or accessibility workflows:
    take an audio recording, transcribe it, then re-speak it
    in a different voice.

    Args:
        audio_path: Path to the input audio file.
        asr_engine: Initialized ASR engine.
        tts_engine: Initialized TTS engine.
        speaker: TTS speaker to use for synthesis.
        hotwords: Optional ASR hotwords.

    Returns:
        PipelineResult with both transcription and synthesis.
    """
    logger.info("Starting transcribe-and-synth pipeline for: %s", audio_path)

    # Step 1: Transcribe
    transcription = asr_engine.transcribe(audio_path, hotwords=hotwords)
    logger.info(
        "Transcription complete: %d segments, %d chars",
        len(transcription.segments),
        len(transcription.text_only),
    )

    # Step 2: Synthesize
    synthesis = tts_engine.synthesize(transcription.text_only, speaker=speaker)
    logger.info("Synthesis complete: %s speaker", speaker)

    return PipelineResult(
        transcription=transcription,
        synthesis=synthesis,
    )


def transcribe_only(
    audio_path: str,
    asr_engine: ASREngine,
    hotwords: Optional[list[str]] = None,
) -> PipelineResult:
    """Transcribe audio only, without synthesis.

    Args:
        audio_path: Path to the input audio file.
        asr_engine: Initialized ASR engine.
        hotwords: Optional ASR hotwords.

    Returns:
        PipelineResult with transcription only.
    """
    logger.info("Starting transcription-only pipeline for: %s", audio_path)
    transcription = asr_engine.transcribe(audio_path, hotwords=hotwords)
    return PipelineResult(transcription=transcription)


def synth_only(
    text: str,
    tts_engine: TTSEngine,
    speaker: str = "Carter",
) -> PipelineResult:
    """Synthesize speech from text only.

    Args:
        text: Input text to synthesize.
        tts_engine: Initialized TTS engine.
        speaker: TTS speaker to use.

    Returns:
        PipelineResult with synthesis only.
    """
    logger.info("Starting synthesis-only pipeline: %d chars", len(text))
    synthesis = tts_engine.synthesize(text, speaker=speaker)
    return PipelineResult(synthesis=synthesis)


def format_transcription_table(result: TranscriptionResult) -> str:
    """Format a transcription result as a readable text table.

    Args:
        result: TranscriptionResult to format.

    Returns:
        Formatted string with timestamps, speakers, and content.
    """
    lines = []
    lines.append(f"{'Time':>12} │ {'Speaker':>8} │ Content")
    lines.append("─" * 80)
    for seg in result.segments:
        time_str = f"{seg.start:.2f}-{seg.end:.2f}"
        lines.append(f"{time_str:>12} │ Speaker {seg.speaker:>2} │ {seg.content}")
    return "\n".join(lines)
