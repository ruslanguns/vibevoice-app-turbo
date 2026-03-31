# VibeVoice Studio

A clean, reproducible web application that combines **VibeVoice-ASR** (speech-to-text) and **VibeVoice-Realtime-0.5B** (real-time text-to-speech) into a single Gradio interface.

Built with [FastAPI](https://fastapi.tiangolo.com/) + [Gradio](https://gradio.app/) for maximum simplicity and easy deployment.

## What it does

- **Transcribe** audio files (up to 60 min) with speaker diarization and timestamps using VibeVoice-ASR (7B)
- **Synthesize** speech from text in real-time (~300ms latency) using VibeVoice-Realtime-0.5B
- **Voice Lab**: explore different speakers and styles (English + 9 multilingual voices)
- Export transcriptions as JSON or plain text

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (T4 or better recommended)
- ~16 GB VRAM for ASR, ~4 GB for Realtime-TTS
- [FFmpeg](https://ffmpeg.org/) installed (`apt install ffmpeg`)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ruslanguns/vibevoice-app-turbo.git
cd vibevoice-app-turbo
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Download experimental multilingual voices

```bash
bash scripts/download_voices.sh
```

### 5. Launch the application

```bash
python -m vibevoice_studio.main
```

The Gradio UI will open at `http://localhost:7860`.

## Project Structure

```
vibevoice-app-turbo/
├── vibevoice_studio/
│   ├── __init__.py
│   ├── main.py              # Gradio UI entry point
│   ├── asr.py               # VibeVoice-ASR wrapper
│   ├── tts.py               # VibeVoice-Realtime-TTS wrapper
│   ├── pipeline.py          # Combined ASR → process → TTS pipeline
│   └── config.py            # Configuration and constants
├── tests/
│   ├── __init__.py
│   ├── test_asr.py
│   ├── test_tts.py
│   ├── test_pipeline.py
│   └── test_config.py
├── scripts/
│   └── download_voices.sh
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── README.md
├── operations-log.md
└── metrics.md
```

## Configuration

Edit `vibevoice_studio/config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `ASR_MODEL_ID` | `microsoft/VibeVoice-ASR-HF` | HuggingFace model ID for ASR |
| `TTS_MODEL_ID` | `microsoft/VibeVoice-Realtime-0.5B` | HuggingFace model ID for TTS |
| `DEVICE` | `auto` | Device for model loading |
| `SAMPLE_RATE` | `24000` | Audio sample rate |

## Usage Examples

### Transcribe audio

1. Open the **ASR Transcription** tab
2. Upload an audio file (WAV, MP3, FLAC)
3. Optionally provide hotwords for better accuracy
4. Click **Transcribe** to get speaker-diariazed output with timestamps

### Generate speech

1. Open the **TTS Synthesis** tab
2. Type or paste text
3. Select a speaker voice
4. Click **Generate** — audio starts in ~300ms

### Voice Lab

1. Open the **Voice Lab** tab
2. Explore different speakers and languages
3. Compare voice outputs side by side

## Models

| Model | Size | Purpose | HuggingFace |
|-------|------|---------|-------------|
| VibeVoice-ASR-HF | 7B | Speech-to-text, diarization | [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) |
| VibeVoice-Realtime-0.5B | 0.5B | Real-time text-to-speech | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) |

> **Note:** VibeVoice-TTS (1.5B) has been disabled by Microsoft and is NOT available.

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [VibeVoice](https://github.com/microsoft/VibeVoice) by Microsoft Research
- [Gradio](https://gradio.app/) for the UI framework
- [Transformers](https://huggingface.co/docs/transformers) by HuggingFace
