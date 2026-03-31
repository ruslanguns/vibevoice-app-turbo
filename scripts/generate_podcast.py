#!/usr/bin/env python3
"""
Generate a 2-speaker podcast episode using VibeVoice-Realtime-0.5B.

Usage:
    python scripts/generate_podcast.py

Output:
    podcast_merged.wav
"""

import os
import glob
import copy
import time
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

SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Podcast script: "Tech Pulse" — eBPF in Production
# ---------------------------------------------------------------------------

PODCAST_SCRIPT = [
    ("host", "Welcome to Tech Pulse, the podcast where we dive into infrastructure tools that are reshaping how we build and run software. Today we are talking about eBPF."),
    ("host", "If you have not heard of eBPF yet, you will. It is one of the biggest shifts in Linux systems programming in the last decade. It lets you run sandboxed programs inside the Linux kernel without modifying kernel source code."),
    ("guest", "That is exactly right. What makes eBPF so powerful is that it lets you observe, filter, and modify network packets, system calls, and kernel functions with minimal overhead. We are talking about nanosecond level hooks."),
    ("host", "So where is eBPF actually being used in production right now?"),
    ("guest", "The biggest area is networking and security. Cilium uses eBPF to replace iptables for Kubernetes networking. Companies like Google and Netflix run it at massive scale."),
    ("host", "What about observability? I keep hearing about Pixie and Hubble."),
    ("guest", "eBPF lets you trace applications without any code changes. No sidecars, no instrumentation. Pixie can auto telemetry your entire cluster with zero app changes."),
    ("host", "What about the challenges?"),
    ("guest", "The learning curve is steep. Writing eBPF requires kernel internals knowledge. The verifier can be frustrating. Debugging is harder than regular code. Tooling has improved but it is still not enough."),
    ("host", "What is your recommendation for teams getting started?"),
    ("guest", "Start with tools, not raw code. Cilium for networking, Tetragon for security, bpftrace for investigations. Then write custom programs only if needed."),
    ("host", "What is next for eBPF?"),
    ("guest", "Programmable networking and AI infrastructure. Optimizing GPU communication, smarter load balancers, custom congestion control. The ecosystem is exploding."),
    ("host", "Thank you for joining us. Check the show notes for links. Until next time, keep shipping."),
]

SPEAKERS = {
    "host": "wayneroger",
    "guest": "de-spk0_man",
}


def find_voices() -> dict:
    """Find available voice .pt files."""
    import vibevoice
    pkg_dir = os.path.dirname(vibevoice.__file__)
    voices_dir = os.path.join(pkg_dir, "..", "demo", "voices", "streaming_model")

    voices = {}
    if os.path.exists(voices_dir):
        for pt_file in glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True):
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            voices[name] = os.path.abspath(pt_file)
    return voices


def synthesize(model, processor, text: str, voice_path: str, device: str, cfg_scale: float = 1.5) -> np.ndarray:
    """Synthesize a single segment using the real VibeVoice API."""
    all_prefilled = torch.load(voice_path, map_location=device, weights_only=False)

    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(all_prefilled) if all_prefilled else None,
    )

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0]
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        return np.squeeze(audio)
    return np.array([])


def main():
    parser = argparse.ArgumentParser(description="Generate a 2-speaker podcast with VibeVoice")
    parser.add_argument("--model_path", default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn = "flash_attention_2" if device == "cuda" else "sdpa"

    print("=" * 60)
    print("Tech Pulse Podcast Generator")
    print("=" * 60)
    print(f"Model:    {args.model_path}")
    print(f"Device:   {device}")
    print(f"Segments: {len(PODCAST_SCRIPT)}")
    print()

    # Load
    print("Loading model...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device if device in ("cuda", "cpu") else None,
        attn_implementation=attn,
    )
    if device == "mps":
        model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)
    print("Model loaded.\n")

    # Voices
    available = find_voices()
    print(f"Available voices: {list(available.keys())}")

    voice_map = {}
    for role, speaker_name in SPEAKERS.items():
        matched = None
        for vname, vpath in available.items():
            if speaker_name in vname or vname in speaker_name:
                matched = vpath
                break
        if not matched and available:
            matched = list(available.values())[0]
            print(f"  Warning: no voice for '{speaker_name}', using fallback")
        voice_map[role] = matched
        print(f"  {role}: {speaker_name} -> {os.path.basename(matched) if matched else 'NONE'}")

    # Generate
    segments = []
    pause = np.zeros(SAMPLE_RATE)  # 1s pause

    total_start = time.time()

    for i, (speaker, text) in enumerate(PODCAST_SCRIPT):
        print(f"\n[{i+1}/{len(PODCAST_SCRIPT)}] {speaker.upper()}: {text[:60]}...")
        t0 = time.time()
        audio = synthesize(model, processor, text, voice_map[speaker], device, args.cfg_scale)
        dt = time.time() - t0

        if len(audio) > 0:
            segments.append(audio)
            segments.append(pause)
            dur = len(audio) / SAMPLE_RATE
            print(f"  OK: {dur:.1f}s audio in {dt:.1f}s (RTF {dt/dur:.2f}x)")
        else:
            print(f"  WARNING: empty audio")

    # Merge
    merged = np.concatenate(segments)
    output_path = str(output_dir / "podcast_merged.wav")
    sf.write(output_path, merged, SAMPLE_RATE)

    total_time = time.time() - total_start
    total_dur = len(merged) / SAMPLE_RATE

    print(f"\n{'=' * 60}")
    print(f"Podcast: {output_path}")
    print(f"Duration: {total_dur:.1f}s ({total_dur/60:.1f}min)")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
