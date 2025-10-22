#!/usr/bin/env python3
"""
MindRoot SIP Plugin - Main Module

Provides SIP phone integration with MindRoot's AI agent system.
Enables voice conversations through SIP protocols with real-time transcription and TTS.

This refactored version imports commands and services from separate modules
for better maintainability and testing.
"""

import logging
import subprocess
import os
import time
from pathlib import Path

# Import commands and services for plugin registration
from .commands import *
from .services import *

# Plugin initialization
logger = logging.getLogger(__name__)

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
        logger.info("JACK daemon is already running")
        with open(log_file, 'a') as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - JACK already running, skipping startup\n")
        return True
    
    try:
        logger.info(f"Starting JACK daemon using script: {script_path}")
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
            logger.info("JACK daemon started successfully")
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
logger.info("MindRoot SIP plugin initializing...")
jack_started = start_jack_daemon()

if jack_started:
    logger.info("MindRoot SIP plugin loaded with JACK audio support")
else:
    logger.warning("MindRoot SIP plugin loaded but JACK daemon may not be running")
    
logger.info("Available commands: call, hangup")
logger.info("Available services: dial_service, sip_audio_out_chunk, end_call_service")
logger.info("JACK logs available at: /tmp/mr_sip_logs/jack_startup.log")
