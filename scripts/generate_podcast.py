#!/usr/bin/env python3
"""
Generate a 2-speaker podcast episode using VibeVoice-Realtime-0.5B.

Usage:
    python scripts/generate_podcast.py

Requires:
    - VibeVoice installed: pip install -e ".[streamingtts]"
      (from https://github.com/microsoft/VibeVoice)
    - NVIDIA GPU with CUDA
    - ~4 GB VRAM

Output:
    - podcast_speaker1.wav  (Host voice)
    - podcast_speaker2.wav  (Guest voice)
    - podcast_merged.wav    (Both speakers interleaved)
"""

import os
import sys
import wave
import struct
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Podcast script: "Tech Pulse" — eBPF in Production (2026)
# ---------------------------------------------------------------------------

PODCAST_SCRIPT = [
    # (speaker, text)
    ("host", "Welcome to Tech Pulse, the podcast where we dive into the infrastructure tools that are quietly reshaping how we build and run software. I'm your host, and today we're talking about eBPF."),
    ("host", "If you haven't heard of eBPF yet, you will. It's one of the biggest shifts in Linux systems programming in the last decade. Think of it as a way to run sandboxed programs inside the Linux kernel, without modifying kernel source code or loading modules."),
    ("guest", "That's exactly right. And what makes eBPF so powerful is that it lets you observe, filter, and even modify network packets, system calls, and kernel functions, all with minimal overhead. We're talking about nanosecond-level hooks."),
    ("host", "So give us the big picture. Where is eBPF actually being used in production right now?"),
    ("guest", "The biggest adoption area is networking and security. Cilium, which is a CNCF graduated project, uses eBPF to replace iptables for Kubernetes networking. Companies like Google, Meta, and Netflix are running it in production at massive scale."),
    ("host", "And what about observability? I keep hearing about tools like Pixie and Hubble."),
    ("guest", "Exactly. eBPF lets you trace applications without any code changes. You don't need to add instrumentation or sidecars. Pixie, which is now part of the Kubernetes ecosystem, can auto-telemetry your entire cluster with zero application changes. That's a game changer for DevOps teams."),
    ("host", "Now let's talk about the challenges. It can't all be perfect, right?"),
    ("guest", "Right. The learning curve is steep. Writing eBPF programs requires knowledge of kernel internals. The verifier can be frustrating. And debugging eBPF programs is significantly harder than regular userspace code. Tooling has improved a lot, but it's still not where it needs to be."),
    ("host", "What about portability? Can I write an eBPF program once and run it anywhere?"),
    ("guest", "That's getting better with CO-RE, which stands for Compile Once, Run Everywhere. Libraries like libbpf make it possible. But you still need to be aware of kernel version differences. A program that works on kernel five fifteen might not work on five four without adjustments."),
    ("host", "So what's your recommendation for teams getting started?"),
    ("guest", "Start with the tools, not the raw eBPF code. Use Cilium for networking, use Tetragon for security observability, use bpftrace for one-liner investigations. Once you understand what these tools can do, then you can start writing custom programs if needed."),
    ("host", "And what's coming next for eBPF?"),
    ("guest", "The really exciting stuff is in programmable networking and AI infrastructure. We're seeing eBPF being used to optimize GPU communication in data centers, to build smarter load balancers, and even to implement custom congestion control. The ecosystem is exploding."),
    ("host", "That's fascinating. Thank you for joining us today. For our listeners, check the show notes for links to Cilium, Tetragon, and the eBPF documentation. Until next time, keep shipping."),
]

# ---------------------------------------------------------------------------
# Speaker configuration
# ---------------------------------------------------------------------------

SPEAKERS = {
    "host": {
        "speaker_name": "Carter",       # Default English male voice
        "output_file": "podcast_speaker1.wav",
    },
    "guest": {
        "speaker_name": "Chloe",         # Experimental English female voice
        "output_file": "podcast_speaker2.wav",
    },
}


def generate_audio_for_text(model_path: str, text: str, speaker_name: str, output_path: str):
    """Generate audio for a single text segment using VibeVoice-Realtime."""
    from vibevoice import RealtimeTTSModel  # type: ignore

    print(f"  Generating with speaker '{speaker_name}': {text[:60]}...")

    model = RealtimeTTSModel.from_pretrained(model_path)
    audio = model.synthesize(text, speaker=speaker_name)

    # Save as WAV
    import numpy as np
    import soundfile as sf
    sf.write(output_path, audio, 24000)
    print(f"  ✅ Saved: {output_path}")
    return output_path


def merge_wav_files(files_with_timing, output_path: str):
    """Merge multiple WAV files into one, with pauses between speakers."""
    import numpy as np
    import soundfile as sf

    print(f"\n Merging {len(files_with_timing)} segments...")
    merged = []
    pause = np.zeros(24000 * 1)  # 1 second pause between segments

    for filepath, speaker in files_with_timing:
        data, sr = sf.read(filepath)
        merged.append(data)
        merged.append(pause)
        # Clean up individual segment
        os.remove(filepath)

    merged = np.concatenate(merged)
    sf.write(output_path, merged, 24000)
    print(f"  ✅ Merged podcast: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a 2-speaker podcast with VibeVoice")
    parser.add_argument(
        "--model_path",
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for output WAV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎙️  Tech Pulse Podcast Generator")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Segments: {len(PODCAST_SCRIPT)}")
    print()

    # Check if experimental voices are available
    print("Note: Make sure to download experimental voices for speaker variety:")
    print("  bash demo/download_experimental_voices.sh")
    print()

    segments = []

    for i, (speaker, text) in enumerate(PODCAST_SCRIPT):
        config = SPEAKERS[speaker]
        seg_path = str(output_dir / f"seg_{i:03d}.wav")
        print(f"[{i+1}/{len(PODCAST_SCRIPT)}] {speaker.upper()}:")
        generate_audio_for_text(args.model_path, text, config["speaker_name"], seg_path)
        segments.append((seg_path, speaker))

    # Merge all segments
    merge_wav_files(segments, str(output_dir / "podcast_merged.wav"))

    print("\n" + "=" * 60)
    print("🎉 Podcast generated successfully!")
    print(f"   Output: {output_dir / 'podcast_merged.wav'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
