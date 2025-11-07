#!/usr/bin/env python3
"""
PySIP Migration Verification Script

This script verifies that the PySIP migration is correctly installed and configured.
Run this before attempting to make calls with the new system.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_color(success):
    """Return color code for success/failure."""
    if success:
        return "\033[92m"  # Green
    else:
        return "\033[91m"  # Red

def reset_color():
    """Return color reset code."""
    return "\033[0m"

def print_status(message, success):
    """Print a status message with color."""
    color = check_color(success)
    status = "✓" if success else "✗"
    print(f"{color}{status}{reset_color()} {message}")

def check_python_version():
    """Check Python version is 3.8+."""
    version = sys.version_info
    success = version.major == 3 and version.minor >= 8
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}", success)
    if not success:
        print("  ⚠ Python 3.8+ required")
    return success

def check_module(module_name, min_version=None):
    """Check if a module is installed and optionally check version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # Simple version comparison (works for most cases)
            installed = tuple(map(int, version.split('.')[:2]))
            required = tuple(map(int, min_version.split('.')[:2]))
            success = installed >= required
            print_status(f"{module_name}: {version} (required: >={min_version})", success)
        else:
            print_status(f"{module_name}: {version}", True)
            success = True
        
        return success
    except ImportError:
        print_status(f"{module_name}: NOT INSTALLED", False)
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    success = path.exists()
    print_status(f"{description}: {filepath}", success)
    return success

def check_backup_exists():
    """Check if backup files exist."""
    backup_dir = Path("/xfiles/update_plugins/mr_sip/src/mr_sip/backup_baresip_s2s")
    if not backup_dir.exists():
        print_status("Backup directory exists", False)
        return False
    
    files = [
        "sip_client_s2s.py.baresip.bak",
        "services_s2s.py.baresip.bak",
        "sip_manager.py.bak"
    ]
    
    all_exist = True
    for filename in files:
        filepath = backup_dir / filename
        exists = filepath.exists()
        if not exists:
            all_exist = False
        print_status(f"  Backup: {filename}", exists)
    
    return all_exist

def check_new_implementation():
    """Check that new implementation files are in place."""
    base_path = Path("/xfiles/update_plugins/mr_sip/src/mr_sip")
    
    # Check sip_client_s2s.py contains PySIP imports
    client_file = base_path / "sip_client_s2s.py"
    if not client_file.exists():
        print_status("sip_client_s2s.py exists", False)
        return False
    
    content = client_file.read_text()
    has_pysip = "from PySIP import SipCall" in content
    no_baresip = "from baresipy import BareSIP" not in content
    no_jack = "JACKAudioCapture" not in content
    has_send_tts = "async def send_tts_audio" in content
    
    print_status("  Contains PySIP imports", has_pysip)
    print_status("  No BareSIP imports", no_baresip)
    print_status("  No JACK imports", no_jack)
    print_status("  Has send_tts_audio() method", has_send_tts)
    
    return has_pysip and no_baresip and no_jack and has_send_tts

def check_env_config():
    """Check environment configuration."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        'SIP_GATEWAY',
        'SIP_USER',
        'SIP_PASSWORD'
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        is_set = value is not None and value != '' and 'no' not in value.lower()
        print_status(f"  {var} configured", is_set)
        if not is_set:
            all_set = False
    
    return all_set

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("PySIP S2S Migration Verification")
    print("="*60 + "\n")
    
    checks = []
    
    # Python version
    print("\n1. Python Environment")
    print("-" * 40)
    checks.append(check_python_version())
    
    # Required modules
    print("\n2. Required Modules")
    print("-" * 40)
    checks.append(check_module('PySIP', '1.8.0'))
    checks.append(check_module('asyncio'))
    checks.append(check_module('queue'))
    checks.append(check_module('dotenv'))
    
    # Optional modules (for other modes)
    print("\n3. Optional Modules (for non-S2S modes)")
    print("-" * 40)
    check_module('baresipy', '0.1.0')
    check_module('jack')
    
    # File structure
    print("\n4. File Structure")
    print("-" * 40)
    checks.append(check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py",
        "S2S Client"
    ))
    checks.append(check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/services_s2s.py",
        "S2S Services"
    ))
    checks.append(check_file_exists(
        "/xfiles/update_plugins/mr_sip/src/mr_sip/sip_manager.py",
        "Session Manager"
    ))
    
    # Backups
    print("\n5. Backup Files")
    print("-" * 40)
    checks.append(check_backup_exists())
    
    # Implementation
    print("\n6. Implementation Verification")
    print("-" * 40)
    checks.append(check_new_implementation())
    
    # Configuration
    print("\n7. Environment Configuration")
    print("-" * 40)
    checks.append(check_env_config())
    
    # Summary
    print("\n" + "="*60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"{check_color(True)}✓ All checks passed ({passed}/{total}){reset_color()}")
        print("\nThe PySIP migration is ready for testing!")
        print("\nNext steps:")
        print("  1. Review TESTING_GUIDE.md")
        print("  2. Run a test call")
        print("  3. Monitor logs for any issues")
        return 0
    else:
        print(f"{check_color(False)}✗ Some checks failed ({passed}/{total}){reset_color()}")
        print("\nPlease address the failed checks before testing.")
        print("\nCommon fixes:")
        print("  - Install PySIP: pip install PySIP>=1.8.0")
        print("  - Configure .env file with SIP credentials")
        print("  - Verify file paths are correct")
        return 1

if __name__ == "__main__":
    sys.exit(main())
