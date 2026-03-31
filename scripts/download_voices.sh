#!/usr/bin/env bash
# Download experimental multilingual voices for VibeVoice-Realtime-0.5B
# These add 9 language voices + 11 English style voices

set -euo pipefail

VOICES_DIR="speakers"
mkdir -p "$VOICES_DIR"

echo "Downloading experimental voices..."
echo "Source: https://github.com/microsoft/VibeVoice"

# Clone or pull the VibeVoice repo for voice files
if [ -d "/tmp/VibeVoice" ]; then
    echo "Updating existing VibeVoice clone..."
    cd /tmp/VibeVoice && git pull --quiet
else
    echo "Cloning VibeVoice repo..."
    git clone --depth 1 https://github.com/microsoft/VibeVoice.git /tmp/VibeVoice
fi

# Copy speaker files
VOICE_SRC="/tmp/VibeVoice/demo/speakers"
if [ -d "$VOICE_SRC" ]; then
    cp -v "$VOICE_SRC"/*.pt "$VOICES_DIR/" 2>/dev/null || echo "No .pt voice files found in $VOICE_SRC"
    echo ""
    echo "Downloaded voices:"
    ls -la "$VOICES_DIR/"
else
    echo "Warning: No speakers directory found at $VOICE_SRC"
    echo "Experimental voices may not be available in this version."
fi

echo ""
echo "Done! Restart the app to detect new voices."
