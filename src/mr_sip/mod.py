#!/usr/bin/env python3
"""
MindRoot SIP Plugin - Main Module

Provides SIP phone integration with MindRoot's AI agent system.
Supports multiple modes:
- Deepgram + separate TTS (v1)
- Deepgram Flux + separate TTS (v2)
- Speech-to-Speech mode (s2s) - for OpenAI Realtime API or similar

This refactored version imports commands and services from separate modules
for better maintainability and testing.
"""

import logging
import subprocess
import os
import time
from pathlib import Path

# Configure logging for entire mr_sip module - CRITICAL only
# This affects all loggers in the mr_sip.* namespace
#logging.getLogger('mr_sip').setLevel(logging.CRITICAL)
logging.getLogger('mr_sip').setLevel(logging.DEBUG)


# Initialize this module's logger
logger = logging.getLogger(__name__)
# Level already set by parent logger above

# Import commands (provider-agnostic)
from .commands import *

# Determine which mode to use
SIP_PROVIDER = os.getenv('SIP_PROVIDER', 'deepgram').lower()

# Import the appropriate service implementation
if SIP_PROVIDER == 's2s':
    from .services_s2s import *
elif SIP_PROVIDER == 'deepgram_v2' or os.getenv('SIP_USE_V2', 'true').lower() in ('true', '1', 'yes', 'on'):
    from .services_v2 import *
else:
    from .services import *

def check_jack_running():
    """Check if JACK daemon is already running."""
    try:
        result = subprocess.run(['pgrep', '-x', 'jackd'], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking JACK status: {e}")
        return False

def start_jack_daemon():
    """Start JACK daemon if not already running."""
    plugin_dir = Path(__file__).parent
    script_path = plugin_dir / "start_jack_daemon.sh"
    log_dir = Path("/tmp/mr_sip_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "jack_startup.log"
    
    if check_jack_running():
        with open(log_file, 'a') as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - JACK already running, skipping startup\n")
        return True
    
    try:
        with open(log_file, 'a') as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting JACK daemon\n")
            f.write(f"Script path: {script_path}\n")
            
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the script
        result = subprocess.run(
            [str(script_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        with open(log_file, 'a') as f:
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")
        
        # Wait a moment and verify JACK started
        time.sleep(2)
        if check_jack_running():
            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - JACK daemon started successfully\n")
            return True
        else:
            logger.error("JACK daemon failed to start")
            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: JACK daemon failed to start\n")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("JACK daemon startup timed out")
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: JACK startup timed out\n")
        return False
    except Exception as e:
        logger.error(f"Error starting JACK daemon: {e}")
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR: {e}\n")
        return False

# Start JACK daemon on plugin load
#jack_started = start_jack_daemon()
jack_started=True
