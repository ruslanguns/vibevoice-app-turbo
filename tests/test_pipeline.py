"""Tests for the pipeline module."""

from __future__ import annotations

from vibevoice_studio.asr import TranscriptionResult, TranscriptionSegment
from vibevoice_studio.pipeline import format_transcription_table


class TestFormatTranscriptionTable:
    """Tests for the format_transcription_table helper."""

    def _make_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(0.0, 5.5, 0, "Hello everyone"),
                TranscriptionSegment(5.5, 10.0, 1, "Welcome to the show"),
            ],
            raw_output="mock raw",
            text_only="Hello everyone. Welcome to the show.",
        )

    def test_table_contains_speakers(self) -> None:
        result = self._make_result()
        table = format_transcription_table(result)
        assert "Speaker" in table
        assert "Speaker  0" in table
        assert "Speaker  1" in table

    def test_table_contains_content(self) -> None:
        result = self._make_result()
        table = format_transcription_table(result)
        assert "Hello everyone" in table
        assert "Welcome to the show" in table

    def test_table_contains_timestamps(self) -> None:
        result = self._make_result()
        table = format_transcription_table(result)
        assert "0.00-5.50" in table
        assert "5.50-10.00" in table

    def test_empty_segments(self) -> None:
        result = TranscriptionResult(
            segments=[],
            raw_output="",
            text_only="",
        )
        table = format_transcription_table(result)
        assert "Time" in table
        assert "Speaker" in table
