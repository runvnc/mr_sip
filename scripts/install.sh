#!/bin/bash
# Install MindRoot SIP plugin with JACK support

set -e

echo "Installing MindRoot SIP Plugin with JACK Audio Support"
echo "====================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

print_status "Starting installation..."

# Update package list
print_status "Updating package list..."
sudo apt-get update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get install -y \
    jackd2 \
    jack-tools \
    libsndfile1-dev \
    python3-pip \
    python3-dev \
    build-essential

print_success "System dependencies installed"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip

# Install the plugin
print_status "Installing MindRoot SIP plugin..."
cd "$(dirname "$0")/.."
pip install -e .

print_success "Python dependencies installed"

# Setup user permissions
print_status "Setting up user permissions..."
if ! groups $USER | grep -q "\baudio\b"; then
    sudo usermod -aG audio $USER
    print_success "User $USER added to audio group"
    NEED_LOGOUT=true
else
    print_success "User $USER is already in audio group"
    NEED_LOGOUT=false
fi

# Create realtime limits
print_status "Setting up realtime audio limits..."
if [ ! -f /etc/security/limits.d/audio.conf ]; then
    sudo tee /etc/security/limits.d/audio.conf << EOF
@audio   -  rtprio     95
@audio   -  memlock    unlimited
EOF
    print_success "Realtime audio limits configured"
else
    print_success "Realtime audio limits already configured"
fi

# Setup baresip configuration
print_status "Setting up baresip configuration..."
BARESIP_DIR="$HOME/.baresip"
BARESIPY_DIR="$HOME/.baresipy"

# Create baresip directories if they don't exist
mkdir -p "$BARESIP_DIR"
mkdir -p "$BARESIPY_DIR"

# Copy configuration template if config doesn't exist
if [ ! -f "$BARESIP_DIR/config" ]; then
    cp config/baresip_config_template "$BARESIP_DIR/config"
    print_success "Baresip configuration installed to $BARESIP_DIR/config"
else
    print_warning "Baresip config already exists at $BARESIP_DIR/config"
    print_warning "Template available at config/baresip_config_template"
fi

# Also copy to baresipy directory
if [ ! -f "$BARESIPY_DIR/config" ]; then
    cp config/baresip_config_template "$BARESIPY_DIR/config"
    print_success "Baresip configuration installed to $BARESIPY_DIR/config"
fi

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/*.sh
print_success "Scripts are now executable"

# Test Python import
print_status "Testing plugin installation..."
if python3 -c "import mr_sip; print('MindRoot SIP plugin imported successfully')" 2>/dev/null; then
    print_success "Plugin installation verified"
else
    print_error "Plugin installation failed - import test failed"
    exit 1
fi

echo ""
print_success "Installation completed successfully!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Set up environment variables in your shell profile:"
echo "   export SIP_GATEWAY='your.sip.provider.com'"
echo "   export SIP_USER='your_sip_username'"
echo "   export SIP_PASSWORD='your_sip_password'"
echo "   export ELEVENLABS_API_KEY='your_elevenlabs_key'"
echo ""
echo "2. Start JACK server:"
echo "   ./scripts/setup_jack.sh"
echo ""
echo "3. Test the plugin:"
echo "   # In MindRoot chat:"
echo "   {\"call\": {\"destination\": \"your_phone_number\"}}"
echo ""
echo "Configuration files:"
echo "• Baresip config: $BARESIP_DIR/config"
echo "• Audio recordings: $BARESIP_DIR/"
echo "• JACK setup script: ./scripts/setup_jack.sh"
echo ""
echo "Troubleshooting:"
echo "• Check JACK status: jack_lsp"
echo "• View plugin logs: Check MindRoot logs for 'mr_sip' entries"
echo "• Test JACK connection: ./scripts/test_jack.sh (if available)"

if [ "$NEED_LOGOUT" = true ]; then
    echo ""
    print_warning "IMPORTANT: You were added to the audio group."
    print_warning "Please log out and back in for changes to take effect."
fi

echo ""
print_success "Installation complete! 🎉"
