"""Tests for the ASR engine module."""

from __future__ import annotations

import pytest

from vibevoice_studio.asr import ASREngine, TranscriptionResult, TranscriptionSegment


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_create_segment(self) -> None:
        seg = TranscriptionSegment(
            start=0.0, end=5.5, speaker=0, content="Hello world"
        )
        assert seg.start == 0.0
        assert seg.end == 5.5
        assert seg.speaker == 0
        assert seg.content == "Hello world"

    def test_to_dict(self) -> None:
        seg = TranscriptionSegment(
            start=1.0, end=2.0, speaker=1, content="Test"
        )
        d = seg.to_dict()
        assert d == {
            "Start": 1.0,
            "End": 2.0,
            "Speaker": 1,
            "Content": "Test",
        }


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def _make_result(self) -> TranscriptionResult:
        segments = [
            TranscriptionSegment(0.0, 5.0, 0, "Hello"),
            TranscriptionSegment(5.0, 10.0, 1, "World"),
        ]
        return TranscriptionResult(
            segments=segments,
            raw_output='[{"Start":0,"End":5,"Speaker":0,"Content":"Hello"}]',
            text_only="Hello World",
        )

    def test_to_json(self) -> None:
        result = self._make_result()
        json_out = result.to_json()
        assert len(json_out) == 2
        assert json_out[0]["Content"] == "Hello"
        assert json_out[1]["Speaker"] == 1

    def test_to_text(self) -> None:
        result = self._make_result()
        assert result.to_text() == "Hello World"


class TestASREngine:
    """Tests for ASREngine class."""

    def test_init_defaults(self) -> None:
        engine = ASREngine()
        assert engine.model_id == "microsoft/VibeVoice-ASR-HF"
        assert engine.device == "auto"
        assert not engine.is_loaded

    def test_init_custom(self) -> None:
        engine = ASREngine(
            model_id="custom/model",
            device="cuda:0",
            dtype="bfloat16",
        )
        assert engine.model_id == "custom/model"
        assert engine.device == "cuda:0"
        assert engine.dtype == "bfloat16"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBEVOICE_DEVICE", "cpu")
        monkeypatch.setenv("VIBEVOICE_DTYPE", "float32")
        # Re-import to pick up env vars
        import importlib
        from vibevoice_studio import config
        importlib.reload(config)
        engine = ASREngine(
            model_id="test/model",
            device=config.DEVICE,
            dtype=config.DTYPE,
        )
        assert engine.device == "cpu"
        assert engine.dtype == "float32"
