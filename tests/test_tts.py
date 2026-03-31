"""Tests for the TTS engine module."""

from __future__ import annotations

import pytest

from vibevoice_studio.tts import TTSEngine, TTSResult, TTSSpeaker


class TestTTSSpeaker:
    """Tests for TTSSpeaker dataclass."""

    def test_create_speaker(self) -> None:
        speaker = TTSSpeaker(name="Carter", language="English")
        assert speaker.name == "Carter"
        assert speaker.language == "English"
        assert not speaker.is_experimental

    def test_experimental_speaker(self) -> None:
        speaker = TTSSpeaker(
            name="spanish_female",
            language="Spanish",
            is_experimental=True,
        )
        assert speaker.is_experimental


class TestTTSResult:
    """Tests for TTSResult dataclass."""

    def test_create_result(self) -> None:
        import numpy as np
        audio = np.zeros(24000)
        result = TTSResult(
            audio=audio,
            sample_rate=24000,
            speaker="Carter",
        )
        assert result.sample_rate == 24000
        assert result.speaker == "Carter"
        assert len(result.audio) == 24000


class TestTTSEngine:
    """Tests for TTSEngine class."""

    def test_init_defaults(self) -> None:
        engine = TTSEngine()
        assert engine.model_path == "microsoft/VibeVoice-Realtime-0.5B"
        assert engine.device == "auto"
        assert not engine.is_loaded

    def test_init_custom(self) -> None:
        engine = TTSEngine(
            model_path="custom/model",
            device="cuda:0",
            dtype="bfloat16",
        )
        assert engine.model_path == "custom/model"
        assert engine.device == "cuda:0"

    def test_get_available_speakers_default(self) -> None:
        speakers = TTSEngine.get_available_speakers()
        assert len(speakers) >= 1
        assert speakers[0].name == "Carter"
        assert speakers[0].language == "English"

    def test_synthesize_raises_without_load(self) -> None:
        engine = TTSEngine()
        with pytest.raises((AssertionError, AttributeError)):
            engine.synthesize("test")
