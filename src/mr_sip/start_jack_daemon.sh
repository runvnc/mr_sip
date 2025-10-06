#!/bin/bash
# JACK Audio Daemon Startup Script for MindRoot SIP
# Configured for 8000 Hz telephony with S16 format

# Kill any existing JACK instances
echo "Stopping any existing JACK instances..."
killall -9 jackd 2>/dev/null
sleep 1

# JACK Configuration
SAMPLE_RATE=8000      # Telephony standard
BUFFER_SIZE=1024      # Period size (adjust for latency vs stability)
NUM_PERIODS=2         # Number of periods per buffer
AUDIO_DEVICE="hw:0"   # Default audio device (change if needed)

echo "Starting JACK daemon..."
echo "Sample Rate: ${SAMPLE_RATE} Hz"
echo "Buffer Size: ${BUFFER_SIZE} frames"
echo "Audio Device: ${AUDIO_DEVICE}"

# Start JACK with ALSA driver
# -d alsa: Use ALSA driver
# -r 8000: Sample rate for telephony
# -p 1024: Period size (buffer size)
# -n 2: Number of periods
# -S: Use 16-bit samples (S16) instead of 32-bit float
# -D: Duplex mode (capture + playback)
# -C: Capture device
# -P: Playback device

jackd -d alsa \
  -r ${SAMPLE_RATE} \
  -p ${BUFFER_SIZE} \
  -n ${NUM_PERIODS} \
  -S \
  -D \
  -C${AUDIO_DEVICE} \
  -P${AUDIO_DEVICE} \
  2>&1 | tee /tmp/jackd.log &

JACK_PID=$!

echo "JACK daemon started with PID: ${JACK_PID}"
echo "Log file: /tmp/jackd.log"

# Wait a moment for JACK to initialize
sleep 2

# Check if JACK is running
if ps -p ${JACK_PID} > /dev/null; then
    echo "JACK is running successfully!"
    echo ""
    echo "To check JACK status:"
    echo "  jack_lsp -c          # List ports and connections"
    echo "  jack_bufsize         # Check buffer size"
    echo "  jack_samplerate      # Check sample rate"
    echo ""
    echo "To stop JACK:"
    echo "  killall jackd"
else
    echo "ERROR: JACK failed to start. Check /tmp/jackd.log for details."
    exit 1
fi
