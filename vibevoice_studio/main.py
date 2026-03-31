"""VibeVoice Studio — Main Gradio application entry point.

Launches a web UI with tabs for:
- ASR Transcription (speech-to-text with diarization)
- TTS Synthesis (real-time text-to-speech)
- Voice Lab (explore speakers and languages)
"""

from __future__ import annotations

import json
import logging

import gradio as gr

from . import __version__
from .asr import ASREngine
from .config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE, TTS_DEFAULT_SPEAKER
from .pipeline import format_transcription_table
from .tts import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global engines (lazy-loaded) ───────────────────────────────────────
asr_engine = ASREngine()
tts_engine = TTSEngine()


# ── ASR Handlers ───────────────────────────────────────────────────────

def handle_transcribe(audio_path: str, hotwords: str) -> tuple[str, str]:
    """Handle ASR transcription request."""
    if not audio_path:
        return "Please upload an audio file.", ""

    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None

    try:
        result = asr_engine.transcribe(audio_path, hotwords=hw_list)
        table = format_transcription_table(result)
        json_str = json.dumps(result.to_json(), indent=2, ensure_ascii=False)
        return table, json_str
    except Exception as e:
        logger.exception("Transcription failed")
        return f"Error: {e}", ""


# ── TTS Handlers ───────────────────────────────────────────────────────

def handle_synthesize(text: str, speaker: str) -> tuple[str | None, str]:
    """Handle TTS synthesis request."""
    if not text.strip():
        return None, "Please enter some text."

    try:
        result = tts_engine.synthesize(text.strip(), speaker=speaker)
        output_path = f"/tmp/vibevoice_output_{speaker}.wav"
        result.save(output_path)
        return output_path, f"Audio generated: {speaker} voice"
    except Exception as e:
        logger.exception("Synthesis failed")
        return None, f"Error: {e}"


# ── UI Construction ────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(
        title=f"VibeVoice Studio v{__version__}",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            f"# 🎙️ VibeVoice Studio v{__version__}\n\n"
            "Combine **VibeVoice-ASR** and **VibeVoice-Realtime-TTS** in one interface.\n\n"
            "_Powered by [VibeVoice](https://github.com/microsoft/VibeVoice) by Microsoft Research._"
        )

        with gr.Tabs():
            # ── ASR Tab ─────────────────────────────────────────────
            with gr.Tab("🎤 ASR Transcription"):
                gr.Markdown(
                    "Upload audio (up to 60 min) for speaker-diariazed transcription "
                    "with timestamps. Supports 50+ languages."
                )
                with gr.Row():
                    with gr.Column():
                        asr_audio = gr.Audio(
                            label="Audio Input",
                            type="filepath",
                        )
                        asr_hotwords = gr.Textbox(
                            label="Hotwords (comma-separated, optional)",
                            placeholder="e.g., VibeVoice, Microsoft, HuggingFace",
                        )
                        asr_btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column():
                        asr_table = gr.Textbox(
                            label="Transcription (Formatted)",
                            lines=15,
                        )
                        asr_json = gr.Code(
                            label="Transcription (JSON)",
                            language="json",
                        )

                asr_btn.click(
                    fn=handle_transcribe,
                    inputs=[asr_audio, asr_hotwords],
                    outputs=[asr_table, asr_json],
                )

            # ── TTS Tab ─────────────────────────────────────────────
            with gr.Tab("🔊 TTS Synthesis"):
                gr.Markdown(
                    "Enter text for real-time speech synthesis (~300ms first-audio latency). "
                    "Supports English and 9 experimental languages."
                )
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=5,
                            placeholder="Enter the text you want to convert to speech...",
                        )
                        tts_speaker = gr.Dropdown(
                            label="Speaker",
                            choices=[TTS_DEFAULT_SPEAKER],
                            value=TTS_DEFAULT_SPEAKER,
                            allow_custom_value=True,
                        )
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                    with gr.Column():
                        tts_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                        )
                        tts_status = gr.Textbox(label="Status")

                tts_btn.click(
                    fn=handle_synthesize,
                    inputs=[tts_text, tts_speaker],
                    outputs=[tts_output, tts_status],
                )

            # ── Voice Lab Tab ──────────────────────────────────────
            with gr.Tab("🧪 Voice Lab"):
                gr.Markdown(
                    "Explore different speakers and languages. "
                    "Download experimental voices with `bash scripts/download_voices.sh`."
                )
                lab_text = gr.Textbox(
                    label="Test Text",
                    value="The quick brown fox jumps over the lazy dog. VibeVoice brings real-time voice AI to everyone.",
                    lines=3,
                )
                with gr.Row():
                    lab_speaker = gr.Dropdown(
                        label="Speaker",
                        choices=[TTS_DEFAULT_SPEAKER],
                        value=TTS_DEFAULT_SPEAKER,
                        allow_custom_value=True,
                    )
                    lab_btn = gr.Button("Test Voice", variant="primary")
                    lab_refresh = gr.Button("🔄 Refresh Speakers")
                lab_output = gr.Audio(label="Voice Output", type="filepath")

                def refresh_speakers():
                    try:
                        speakers = TTSEngine.get_available_speakers()
                        choices = [s.name for s in speakers]
                        return gr.update(choices=choices, value=choices[0] if choices else TTS_DEFAULT_SPEAKER)
                    except Exception:
                        return gr.update()

                lab_refresh.click(fn=refresh_speakers, outputs=[lab_speaker])
                lab_btn.click(
                    fn=handle_synthesize,
                    inputs=[lab_text, lab_speaker],
                    outputs=[lab_output, gr.Textbox(visible=False)],
                )

    return app


def main() -> None:
    """Launch the VibeVoice Studio application."""
    logger.info("Starting VibeVoice Studio v%s", __version__)

    app = build_ui()
    app.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
    )


if __name__ == "__main__":
    main()
