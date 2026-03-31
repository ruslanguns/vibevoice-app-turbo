"""Tests for the config module."""

from __future__ import annotations

from vibevoice_studio.config import (
    ASR_MODEL_ID,
    ASR_TOKENIZER_CHUNK_SIZE,
    DEVICE,
    DTYPE,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    MAX_ASR_DURATION_SECONDS,
    SAMPLE_RATE,
    TTS_DEFAULT_SPEAKER,
    TTS_MODEL_ID,
    TTS_SUPPORTED_LANGUAGES,
)


class TestConfigDefaults:
    """Verify config constants have expected values."""

    def test_asr_model_id(self) -> None:
        assert ASR_MODEL_ID == "microsoft/VibeVoice-ASR-HF"

    def test_tts_model_id(self) -> None:
        assert TTS_MODEL_ID == "microsoft/VibeVoice-Realtime-0.5B"

    def test_device(self) -> None:
        assert DEVICE == "auto"

    def test_dtype(self) -> None:
        assert DTYPE == "float16"

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24_000

    def test_max_asr_duration(self) -> None:
        assert MAX_ASR_DURATION_SECONDS == 3600

    def test_asr_chunk_size(self) -> None:
        assert ASR_TOKENIZER_CHUNK_SIZE == 64_000

    def test_tts_default_speaker(self) -> None:
        assert TTS_DEFAULT_SPEAKER == "Carter"

    def test_tts_languages(self) -> None:
        assert len(TTS_SUPPORTED_LANGUAGES) == 10
        assert "English" in TTS_SUPPORTED_LANGUAGES
        assert "Spanish" in TTS_SUPPORTED_LANGUAGES


class TestConfigEnvOverrides:
    """Test that env vars can override config values."""

    def test_env_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBEVOICE_DEVICE", "cpu")
        import importlib
        from vibevoice_studio import config
        importlib.reload(config)
        assert config.DEVICE == "cpu"
        monkeypatch.delenv("VIBEVOICE_DEVICE", raising=False)
        importlib.reload(config)

    def test_env_gradio_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRADIO_SERVER_PORT", "9999")
        import importlib
        from vibevoice_studio import config
        importlib.reload(config)
        assert config.GRADIO_SERVER_PORT == 9999
        monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
        importlib.reload(config)
