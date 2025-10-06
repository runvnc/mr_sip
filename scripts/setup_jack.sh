#!/bin/bash
# Setup JACK server for MindRoot SIP plugin

set -e

echo "Setting up JACK server for MindRoot SIP plugin..."
echo "================================================"

# Check if JACK is installed
if ! command -v jackd &> /dev/null; then
    echo "JACK not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y jackd2 jack-tools
else
    echo "✓ JACK is already installed"
fi

# Add user to audio group if not already
if ! groups $USER | grep -q "\baudio\b"; then
    echo "Adding $USER to audio group..."
    sudo usermod -aG audio $USER
    echo "✓ User added to audio group (logout/login required)"
else
    echo "✓ User is already in audio group"
fi

# Create realtime limits for audio
if [ ! -f /etc/security/limits.d/audio.conf ]; then
    echo "Setting up realtime audio limits..."
    sudo tee /etc/security/limits.d/audio.conf << EOF
@audio   -  rtprio     95
@audio   -  memlock    unlimited
EOF
    echo "✓ Realtime audio limits configured"
else
    echo "✓ Realtime audio limits already configured"
fi

# Kill existing JACK server
echo "Stopping any existing JACK server..."
sudo killall -9 jackd 2>/dev/null || true
sleep 1

# Start JACK with dummy driver at 8000 Hz (telephony rate)
echo "Starting JACK server at 8000 Hz..."
jackd -d dummy -r 8000 &
JACK_PID=$!

# Wait for JACK to start
sleep 2

# Check if JACK is running
if ps -p $JACK_PID > /dev/null; then
    echo "✓ JACK server started successfully (PID: $JACK_PID)"
    echo "✓ Sample rate: 8000 Hz (telephony)"
    echo "✓ Driver: dummy (no hardware conflicts)"
else
    echo "✗ Failed to start JACK server"
    exit 1
fi

# Show available ports
echo ""
echo "Available JACK ports:"
jack_lsp || echo "No ports available yet (normal for fresh start)"

echo ""
echo "JACK setup complete!"
echo "==================="
echo "• JACK server is running at 8000 Hz"
echo "• Use 'jack_lsp' to list available ports"
echo "• Use 'sudo killall jackd' to stop JACK server"
echo "• Baresip will create ports when a call is active"
echo ""
echo "Next steps:"
echo "1. Configure baresip (copy config/baresip_config_template to ~/.baresip/config)"
echo "2. Test SIP plugin with: python -c 'import mr_sip; print(\"Plugin loaded successfully\")'"
echo "3. Make a test call to verify JACK integration"

if ! groups $USER | grep -q "\baudio\b" 2>/dev/null; then
    echo ""
    echo "⚠️  IMPORTANT: You were added to the audio group."
    echo "   Please log out and back in for changes to take effect."
fi
