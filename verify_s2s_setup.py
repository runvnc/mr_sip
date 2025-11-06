#!/usr/bin/env python3
"""
Verification Script for S2S Mode Setup

This script checks that all necessary components are in place for S2S mode.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} MISSING: {filepath}")
        return False

def check_env_var(var_name, required=True):
    """Check if an environment variable is set."""
    value = os.getenv(var_name)
    if value:
        # Mask sensitive values
        if 'KEY' in var_name or 'PASSWORD' in var_name:
            display_value = value[:8] + '...' if len(value) > 8 else '***'
        else:
            display_value = value
        print(f"✓ {var_name}={display_value}")
        return True
    else:
        status = "✗" if required else "○"
        print(f"{status} {var_name} not set" + (" (REQUIRED)" if required else " (optional)"))
        return not required

def main():
    print("="*60)
    print("MindRoot SIP Plugin - S2S Mode Verification")
    print("="*60)
    print()
    
    all_good = True
    
    # Check core files
    print("Checking Core Files:")
    print("-" * 40)
    all_good &= check_file_exists(
        "/files/mindroot/src/mindroot/coreplugins/agent/speech_to_speech.py",
        "SpeechToSpeechAgent"
    )
    all_good &= check_file_exists(
        "/xfiles/plugins_ah/ah_openai/src/ah_openai/speech_to_speech.py",
        "OpenAI S2S Service"
    )
    print()
    
    # Check SIP plugin files
    print("Checking SIP Plugin Files:")
    print("-" * 40)
    all_good &= check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/mod.py",
        "Plugin Module"
    )
    all_good &= check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py",
        "S2S SIP Client"
    )
    all_good &= check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/services_s2s.py",
        "S2S Services"
    )
    all_good &= check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/audio/jack_input_capture.py",
        "JACK Audio Capture"
    )
    print()
    
    # Check documentation
    print("Checking Documentation:")
    print("-" * 40)
    check_file_exists(
        "/xfiles/update_plugins/mr_sip/S2S_SETUP.md",
        "Setup Guide"
    )
    check_file_exists(
        "/xfiles/update_plugins/mr_sip/.env.s2s.example",
        "Example Config"
    )
    check_file_exists(
        "/tmp/s2splan.md",
        "Implementation Plan"
    )
    check_file_exists(
        "/tmp/s2s_implementation_summary.md",
        "Implementation Summary"
    )
    print()
    
    # Check environment variables
    print("Checking Environment Variables:")
    print("-" * 40)
    all_good &= check_env_var("SIP_PROVIDER", required=True)
    all_good &= check_env_var("SIP_GATEWAY", required=True)
    all_good &= check_env_var("SIP_USER", required=True)
    all_good &= check_env_var("SIP_PASSWORD", required=True)
    all_good &= check_env_var("OPENAI_API_KEY", required=True)
    check_env_var("OPENAI_REALTIME_MODEL", required=False)
    check_env_var("OPENAI_VOICE", required=False)
    check_env_var("AUDIO_CAPTURE_METHOD", required=False)
    check_env_var("SIP_CALL_ESTABLISH_TIMEOUT", required=False)
    print()
    
    # Check JACK
    print("Checking JACK Audio:")
    print("-" * 40)
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-x', 'jackd'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("✓ JACK daemon is running")
        else:
            print("✗ JACK daemon is NOT running")
            print("  Run: cd /xfiles/update_plugins/mr_sip/src/mr_sip && ./start_jack_daemon.sh")
            all_good = False
    except Exception as e:
        print(f"✗ Error checking JACK: {e}")
        all_good = False
    print()
    
    # Check Python dependencies
    print("Checking Python Dependencies:")
    print("-" * 40)
    dependencies = [
        'numpy',
        'scipy',
        'jack',
        'websocket',
        'sounddevice',
        'baresipy'
    ]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} NOT INSTALLED")
            all_good = False
    print()
    
    # Summary
    print("="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Your S2S mode setup appears to be complete!")
        print()
        print("Next steps:")
        print("1. Create an agent with agent_class: SpeechToSpeechAgent")
        print("2. Enable 'call' and 'hangup' commands for the agent")
        print("3. Start the agent and test local audio first")
        print("4. Try making a test call")
        print()
        print("See S2S_SETUP.md for detailed instructions.")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please review the errors above and:")
        print("1. Set missing environment variables in .env")
        print("2. Install missing dependencies")
        print("3. Start JACK daemon if needed")
        print()
        print("See S2S_SETUP.md for troubleshooting help.")
    print("="*60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
