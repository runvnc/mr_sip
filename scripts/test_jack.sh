#!/bin/bash
# Test JACK integration for MindRoot SIP plugin

set -e

echo "Testing JACK Integration for MindRoot SIP Plugin"
echo "==============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

TEST_FAILED=false

# Test 1: Check if JACK is installed
print_status "Checking if JACK is installed..."
if command -v jackd &> /dev/null; then
    print_success "JACK is installed"
else
    print_error "JACK is not installed"
    TEST_FAILED=true
fi

# Test 2: Check if user is in audio group
print_status "Checking audio group membership..."
if groups $USER | grep -q "\baudio\b"; then
    print_success "User $USER is in audio group"
else
    print_error "User $USER is not in audio group"
    TEST_FAILED=true
fi

# Test 3: Check if JACK server is running
print_status "Checking if JACK server is running..."
if pgrep -x "jackd" > /dev/null; then
    print_success "JACK server is running"
    
    # Show JACK info
    print_status "JACK server information:"
    jack_sample_rate=$(jack_sample_rate 2>/dev/null || echo "unknown")
    jack_buffer_size=$(jack_buffer_size 2>/dev/null || echo "unknown")
    echo "  Sample rate: $jack_sample_rate Hz"
    echo "  Buffer size: $jack_buffer_size frames"
    
    if [ "$jack_sample_rate" = "8000" ]; then
        print_success "JACK is running at correct sample rate (8000 Hz)"
    else
        print_warning "JACK sample rate is $jack_sample_rate Hz (expected 8000 Hz for telephony)"
    fi
else
    print_error "JACK server is not running"
    print_status "To start JACK server: ./setup_jack.sh"
    TEST_FAILED=true
fi

# Test 4: Check Python dependencies
print_status "Checking Python dependencies..."

# Test JACK-Client
if python3 -c "import jack; print('JACK-Client version:', jack.version)" 2>/dev/null; then
    print_success "JACK-Client Python module is available"
else
    print_error "JACK-Client Python module is not available"
    TEST_FAILED=true
fi

# Test numpy
if python3 -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null; then
    print_success "NumPy is available"
else
    print_error "NumPy is not available"
    TEST_FAILED=true
fi

# Test scipy
if python3 -c "import scipy; print('SciPy version:', scipy.__version__)" 2>/dev/null; then
    print_success "SciPy is available"
else
    print_error "SciPy is not available"
    TEST_FAILED=true
fi

# Test pydub
if python3 -c "import pydub; print('Pydub is available')" 2>/dev/null; then
    print_success "Pydub is available"
else
    print_error "Pydub is not available"
    TEST_FAILED=true
fi

# Test 5: Check MindRoot SIP plugin
print_status "Checking MindRoot SIP plugin..."
if python3 -c "import mr_sip; print('MindRoot SIP plugin loaded successfully')" 2>/dev/null; then
    print_success "MindRoot SIP plugin can be imported"
else
    print_error "MindRoot SIP plugin cannot be imported"
    TEST_FAILED=true
fi

# Test 6: Test JACK client creation
print_status "Testing JACK client creation..."
if pgrep -x "jackd" > /dev/null; then
    cat > /tmp/test_jack_client.py << 'EOF'
import jack
import sys

try:
    client = jack.Client("TestClient")
    client.activate()
    print(f"JACK client created successfully")
    print(f"Sample rate: {client.samplerate} Hz")
    print(f"Block size: {client.blocksize} frames")
    client.deactivate()
    print("JACK client test passed")
except Exception as e:
    print(f"JACK client test failed: {e}")
    sys.exit(1)
EOF

    if python3 /tmp/test_jack_client.py 2>/dev/null; then
        print_success "JACK client creation test passed"
    else
        print_error "JACK client creation test failed"
        TEST_FAILED=true
    fi
    
    rm -f /tmp/test_jack_client.py
else
    print_warning "Skipping JACK client test (JACK server not running)"
fi

# Test 7: Check baresip configuration
print_status "Checking baresip configuration..."
BARESIP_CONFIG="$HOME/.baresip/config"
BARESIPY_CONFIG="$HOME/.baresipy/config"

if [ -f "$BARESIP_CONFIG" ]; then
    print_success "Baresip config found at $BARESIP_CONFIG"
    
    # Check for JACK module
    if grep -q "module.*jack.so" "$BARESIP_CONFIG"; then
        print_success "JACK module is enabled in baresip config"
    else
        print_error "JACK module is not enabled in baresip config"
        TEST_FAILED=true
    fi
    
    # Check for float format
    if grep -q "ausrc_format.*float" "$BARESIP_CONFIG"; then
        print_success "Audio source format is set to float (required for JACK)"
    else
        print_error "Audio source format is not set to float (required for JACK)"
        TEST_FAILED=true
    fi
else
    print_error "Baresip config not found at $BARESIP_CONFIG"
    print_status "Run ./install.sh to create default configuration"
    TEST_FAILED=true
fi

# Test 8: List available JACK ports
if pgrep -x "jackd" > /dev/null; then
    print_status "Available JACK ports:"
    if jack_lsp 2>/dev/null | head -10; then
        echo "  (showing first 10 ports, use 'jack_lsp' to see all)"
    else
        echo "  No JACK ports available (normal if no applications are connected)"
    fi
fi

echo ""
echo "Test Summary"
echo "============"

if [ "$TEST_FAILED" = true ]; then
    print_error "Some tests failed. Please address the issues above."
    echo ""
    echo "Common solutions:"
    echo "• Install missing dependencies: ./install.sh"
    echo "• Start JACK server: ./setup_jack.sh"
    echo "• Add user to audio group: sudo usermod -aG audio \$USER (then logout/login)"
    echo "• Check baresip configuration: copy config/baresip_config_template to ~/.baresip/config"
    exit 1
else
    print_success "All tests passed! JACK integration is ready."
    echo ""
    echo "You can now:"
    echo "• Make SIP calls with: {\"call\": {\"destination\": \"phone_number\"}}"
    echo "• Monitor JACK connections with: jack_lsp -c"
    echo "• View JACK system info with: jack_sample_rate && jack_buffer_size"
    exit 0
fi
