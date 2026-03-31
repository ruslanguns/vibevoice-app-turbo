#!/usr/bin/env python3
"""
Generate a 2-speaker podcast episode using VibeVoice-Realtime-0.5B.

Usage:
    python scripts/generate_podcast.py

Output:
    podcast_merged.wav (both speakers interleaved)
"""

import os
import glob
import argparse
from pathlib import Path

import torch
import soundfile as sf
import numpy as np

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# ---------------------------------------------------------------------------
# Podcast script: "Tech Pulse" — eBPF in Production (2026)
# ---------------------------------------------------------------------------

PODCAST_SCRIPT = [
    ("host", "Welcome to Tech Pulse, the podcast where we dive into the infrastructure tools that are quietly reshaping how we build and run software. I'm your host, and today we're talking about eBPF."),
    ("host", "If you haven't heard of eBPF yet, you will. It's one of the biggest shifts in Linux systems programming in the last decade. Think of it as a way to run sandboxed programs inside the Linux kernel, without modifying kernel source code or loading modules."),
    ("guest", "That's exactly right. And what makes eBPF so powerful is that it lets you observe, filter, and even modify network packets, system calls, and kernel functions, all with minimal overhead. We're talking about nanosecond-level hooks."),
    ("host", "So give us the big picture. Where is eBPF actually being used in production right now?"),
    ("guest", "The biggest adoption area is networking and security. Cilium, which is a CNCF graduated project, uses eBPF to replace iptables for Kubernetes networking. Companies like Google, Meta, and Netflix are running it in production at massive scale."),
    ("host", "And what about observability? I keep hearing about tools like Pixie and Hubble."),
    ("guest", "Exactly. eBPF lets you trace applications without any code changes. You don't need to add instrumentation or sidecars. Pixie can auto-telemetry your entire cluster with zero application changes. That's a game changer for DevOps teams."),
    ("host", "Now let's talk about the challenges. It can't all be perfect, right?"),
    ("guest", "Right. The learning curve is steep. Writing eBPF programs requires knowledge of kernel internals. The verifier can be frustrating. And debugging eBPF programs is significantly harder than regular userspace code. Tooling has improved a lot, but it's still not where it needs to be."),
    ("host", "What about portability? Can I write an eBPF program once and run it anywhere?"),
    ("guest", "That's getting better with CO-RE, which stands for Compile Once, Run Everywhere. Libraries like libbpf make it possible. But you still need to be aware of kernel version differences."),
    ("host", "So what's your recommendation for teams getting started?"),
    ("guest", "Start with the tools, not the raw eBPF code. Use Cilium for networking, use Tetragon for security observability, use bpftrace for one-liner investigations. Once you understand what these tools can do, then you can start writing custom programs if needed."),
    ("host", "And what's coming next for eBPF?"),
    ("guest", "The really exciting stuff is in programmable networking and AI infrastructure. We're seeing eBPF being used to optimize GPU communication in data centers, to build smarter load balancers, and even to implement custom congestion control. The ecosystem is exploding."),
    ("host", "That's fascinating. Thank you for joining us today. For our listeners, check the show notes for links to Cilium, Tetragon, and the eBPF documentation. Until next time, keep shipping."),
]

# Voice mapping
SPEAKERS = {
    "host": "carter",
    "guest": "chloe",
}


def find_voice_path(speaker_name: str, model_dir: str = None) -> str:
    """Find voice prompt file for a speaker."""
    # Check bundled voices first
    search_dirs = []
    if model_dir:
        search_dirs.append(model_dir)

    # Check demo/voices relative to vibevoice package
    import vibevoice
    pkg_dir = os.path.dirname(vibevoice.__file__)
    search_dirs.append(os.path.join(pkg_dir, "..", "demo", "voices", "streaming_model"))
    search_dirs.append(os.path.join(os.path.dirname(__file__), "..", "demo", "voices", "streaming_model"))

    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        pt_files = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
        for pt_file in pt_files:
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            if name == speaker_name.lower():
                return os.path.abspath(pt_file)

    # Return first available voice as fallback
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        pt_files = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
        if pt_files:
            print(f"Warning: No voice for '{speaker_name}', using: {pt_files[0]}")
            return os.path.abspath(pt_files[0])

    raise FileNotFoundError(
        f"No voice files found for '{speaker_name}'. "
        f"Run: bash demo/download_experimental_voices.sh from the VibeVoice repo"
    )


def synthesize_segment(model, processor, text: str, voice_path: str, device: str) -> np.ndarray:
    """Synthesize a single text segment."""
    all_audio = []

    for audio_chunk in model.stream_synthesize(
        text=text,
        processor=processor,
        voice_prompt_path=voice_path,
        device=device,
        cfg_scale=1.5,
    ):
        if audio_chunk is not None:
            all_audio.append(audio_chunk.cpu().numpy().squeeze())

    if all_audio:
        return np.concatenate(all_audio)
    return np.array([])


def main():
    parser = argparse.ArgumentParser(description="Generate a 2-speaker podcast with VibeVoice")
    parser.add_argument("--model_path", default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"

    print("=" * 60)
    print("Tech Pulse Podcast Generator")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Segments: {len(PODCAST_SCRIPT)}")
    print()

    # Load model and processor
    print("Loading model...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=load_dtype,
        device_map=device if device == "cuda" else None,
        attn_implementation=attn_impl,
    )
    if device == "mps":
        model.to("mps")
    elif device == "cpu":
        model.to("cpu")

    model.eval()
    print("Model loaded.\n")

    # Find voices
    voices = {}
    for role, speaker_name in SPEAKERS.items():
        voices[role] = find_voice_path(speaker_name, args.model_path)
        print(f"  {role}: {speaker_name} -> {voices[role]}")

    # Generate segments
    segments = []
    pause = np.zeros(24000)  # 1s pause

    for i, (speaker, text) in enumerate(PODCAST_SCRIPT):
        print(f"\n[{i+1}/{len(PODCAST_SCRIPT)}] {speaker.upper()}: {text[:60]}...")
        audio = synthesize_segment(model, processor, text, voices[speaker], device)
        if len(audio) > 0:
            segments.append(audio)
            segments.append(pause)
            print(f"  OK ({len(audio)/24000:.1f}s)")
        else:
            print(f"  WARNING: empty audio for segment {i+1}")

    # Merge
    merged = np.concatenate(segments)
    output_path = str(output_dir / "podcast_merged.wav")
    sf.write(output_path, merged, 24000)

    duration = len(merged) / 24000
    print(f"\n{'=' * 60}")
    print(f"Podcast saved: {output_path}")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
