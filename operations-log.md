# Operations Log — VibeVoice Studio Build

All operations recorded with ISO 8601 timestamps (UTC).

## Phase 1: Research & Planning

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:14:39Z | `exec` | Created project directory `/tmp/vibevoice-turbo/` |
| 2026-03-31T20:14:41Z | `web_search` | Query: "VibeVoice Microsoft ASR Realtime TTS HuggingFace 2025" — 10 results |
| 2026-03-31T20:14:44Z | `web_fetch` | https://huggingface.co/microsoft/VibeVoice-ASR — ASR model card, 60-min processing, 50+ languages |
| 2026-03-31T20:14:45Z | `web_fetch` | https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B — TTS model card, 0.5B params, ~300ms latency |
| 2026-03-31T20:14:44Z | `web_fetch` | https://huggingface.co/microsoft/VibeVoice-ASR-HF — Transformers v5.3.0 compatible version with code examples |
| 2026-03-31T20:14:45Z | `web_fetch` | https://github.com/microsoft/VibeVoice — Main repo: ASR, Realtime TTS, disabled TTS 1.5B |
| 2026-03-31T20:14:49Z | `web_fetch` | https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md — Installation: `pip install -e .[streamingtts]`, Docker recommended |
| 2026-03-31T20:14:49Z | `web_fetch` | https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md — Installation: `pip install -e .`, `ffmpeg` required |
| 2026-03-31T20:15:00Z | `mcporter call context7` | `resolve-library-id` for FastAPI — found `/fastapi/fastapi` (1679 snippets, score 81.09) |
| 2026-03-31T20:15:10Z | `mcporter call context7` | `resolve-library-id` for Gradio — found `/gradio-app/gradio` (2772 snippets, score 85.77) |
| 2026-03-31T20:15:20Z | `mcporter call context7` | `resolve-library-id` for FastRTC — found `/gradio-app/fastrtc` (368 snippets), relevant for real-time audio |
| 2026-03-31T20:15:40Z | `mcporter call context7` | `query-docs` for FastRTC — got streaming audio API examples, TTS/STT integration patterns |

### Research Decision
**Framework choice: Gradio** — Best fit because:
1. VibeVoice's own demos use Gradio (confirmed in their repo)
2. Native audio components (gr.Audio) for upload/playback
3. No frontend code needed — pure Python
4. FastAPI is used under the hood by Gradio for serving
5. FastRTC is interesting but adds complexity; keeping it simple

**App concept: "VibeVoice Studio"** — A 3-tab Gradio app:
- Tab 1: ASR Transcription (upload audio → get diarized text)
- Tab 2: TTS Synthesis (text → speech with speaker selection)
- Tab 3: Voice Lab (explore speakers/languages)

## Phase 2: Project Setup

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:16:32Z | `exec` | `git init` in `/tmp/vibevoice-turbo/`, branch renamed to `main` |
| 2026-03-31T20:16:40Z | `write` | `.gitignore` — Python, venv, IDE, model artifacts, audio, env |
| 2026-03-31T20:16:45Z | `write` | `README.md` — Full setup guide, project structure, usage examples |
| 2026-03-31T20:16:50Z | `write` | `pyproject.toml` — setuptools, ruff, pytest, mypy config |
| 2026-03-31T20:16:55Z | `write` | `requirements.txt` — Core deps: torch, transformers>=5.3.0, gradio, fastapi |
| 2026-03-31T20:17:00Z | `write` | `LICENSE` — MIT license |
| 2026-03-31T20:17:05Z | `git commit` | "Initial project setup: README, config, and dependencies" |

## Phase 3: Core Modules

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:17:15Z | `write` | `vibevoice_studio/__init__.py` — Package with version |
| 2026-03-31T20:17:20Z | `write` | `vibevoice_studio/config.py` — Model IDs, device, audio, TTS speakers, server config |
| 2026-03-31T20:17:25Z | `git commit` | "Add package init and configuration module" |
| 2026-03-31T20:17:40Z | `write` | `vibevoice_studio/asr.py` — ASREngine with lazy loading, hotwords, parsed/raw/text output |
| 2026-03-31T20:17:45Z | `git commit` | "Add ASR engine wrapper for VibeVoice-ASR" |
| 2026-03-31T20:18:00Z | `write` | `vibevoice_studio/tts.py` — TTSEngine with sync/streaming synthesis, speaker discovery |
| 2026-03-31T20:18:05Z | `git commit` | "Add TTS engine wrapper for VibeVoice-Realtime-0.5B" |
| 2026-03-31T20:18:20Z | `write` | `vibevoice_studio/pipeline.py` — Combined ASR+TTS workflows, table formatting |
| 2026-03-31T20:18:25Z | `git commit` | "Add pipeline module for combined ASR+TTS workflows" |
| 2026-03-31T20:18:40Z | `write` | `vibevoice_studio/main.py` — Full Gradio UI with 3 tabs, lazy engine init |
| 2026-03-31T20:18:45Z | `git commit` | "Add main Gradio UI with ASR, TTS, and Voice Lab tabs" |

## Phase 4: Tests & Scripts

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:19:00Z | `write` | `tests/__init__.py` |
| 2026-03-31T20:19:05Z | `write` | `tests/test_asr.py` — 6 tests: segment creation, to_dict, result JSON/text, engine init |
| 2026-03-31T20:19:10Z | `write` | `tests/test_tts.py` — 5 tests: speaker, result, engine init, speaker discovery |
| 2026-03-31T20:19:15Z | `write` | `tests/test_pipeline.py` — 4 tests: table formatting with various inputs |
| 2026-03-31T20:19:20Z | `write` | `tests/test_config.py` — 11 tests: all defaults + env var overrides |
| 2026-03-31T20:19:25Z | `git commit` | "Add comprehensive test suite" |
| 2026-03-31T20:19:35Z | `write` | `scripts/download_voices.sh` — Download experimental multilingual voices |
| 2026-03-31T20:19:40Z | `git commit` | "Add voice download script for experimental speakers" |

## Phase 5: Validation

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:20:00Z | `exec` | Python syntax check on all modules |
| 2026-03-31T20:20:10Z | `exec` | Import verification for all modules |
| 2026-03-31T20:20:20Z | `exec` | pytest run (unit tests only, no GPU) |
| 2026-03-31T20:20:30Z | `exec` | ruff lint check |

## Phase 6: Documentation & Publishing

| Timestamp | Operation | Details |
|-----------|-----------|---------|
| 2026-03-31T20:20:40Z | `write` | `operations-log.md` — This file |
| 2026-03-31T20:21:00Z | `write` | `metrics.md` — Performance metrics |
| 2026-03-31T20:21:10Z | `git commit` | "Add operations log and metrics" |
| 2026-03-31T20:21:30Z | `exec` | `gh repo create` — Public repo "vibevoice-app-turbo" |
| 2026-03-31T20:21:45Z | `exec` | `git push` — Push all commits to remote |
