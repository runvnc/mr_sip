#!/bin/bash
# JACK Audio Daemon Startup Script for MindRoot SIP
# Configured for 8000 Hz telephony using dummy driver

# Kill any existing JACK instances
echo "Stopping any existing JACK instances..."
killall -9 jackd 2>/dev/null
sleep 1

# JACK Configuration
SAMPLE_RATE=8000       # Telephony standard (resampled to 16000 in capture code)
PERIOD_SIZE=256        # Smaller period = less latency, smoother audio (32ms at 8000 Hz)
WAIT_TIME=32000        # Microseconds between engine processes (match period time)

echo "Starting JACK daemon..."
echo "Sample Rate: ${SAMPLE_RATE} Hz"
echo "Period Size: ${PERIOD_SIZE} frames (${PERIOD_SIZE}/${SAMPLE_RATE} = $(echo "scale=1; ${PERIOD_SIZE}*1000/${SAMPLE_RATE}" | bc)ms)"
echo "Wait Time: ${WAIT_TIME} microseconds"
echo "Driver: dummy (no physical audio device)"

# Start JACK with dummy driver
# -d dummy: Use dummy driver (no physical audio, just routing)
# -r 8000: Sample rate for telephony
# -p 256: Period size (smaller = smoother, less latency)
# -w 32000: Wait time between engine processes

jackd -d dummy \
  -r ${SAMPLE_RATE} \
  -p ${PERIOD_SIZE} \
  -w ${WAIT_TIME} \
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
