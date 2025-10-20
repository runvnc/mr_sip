from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from lib.templates import render
import os
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import pytz
import whisper

router = APIRouter()

@router.get("/calls")
async def list_calls(request: Request):
    """List all call recordings with metadata"""
    calls_dir = Path("data/calls")
    
    # Set timezone to Chicago
    chicago_tz = pytz.timezone('America/Chicago')
    calls = []
    
    if calls_dir.exists():
        for wav_file in sorted(calls_dir.glob("*.wav"), key=os.path.getmtime, reverse=True):
            log_id = wav_file.stem
            
            # Find the chatlog file
            chatlog_path = find_chatlog(log_id)
            phone_number = None
            agent_name = None
            
            if chatlog_path:
                # Extract phone number and agent name
                phone_number = extract_phone_number(chatlog_path)
                agent_name = extract_agent_name(chatlog_path)
            
            # Get file modification time
            mtime_utc = datetime.fromtimestamp(wav_file.stat().st_mtime, tz=pytz.UTC)
            mtime_chicago = mtime_utc.astimezone(chicago_tz)
            mtime = mtime_chicago
            
            calls.append({
                "log_id": log_id,
                "filename": wav_file.name,
                "time": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_name": agent_name or "Unknown",
                "phone_number": phone_number or "Unknown",
                "session_path": f"/session/{agent_name}/{log_id}" if agent_name else None
            })
    
    html = await render('calls', {"calls": calls})
    return HTMLResponse(html)

@router.get("/calls/audio/{log_id}")
async def get_audio(log_id: str):
    """Serve audio file for a call"""
    audio_path = Path(f"data/calls/{log_id}.wav")
    
    if not audio_path.exists():
        return JSONResponse({"error": "Audio file not found"}, status_code=404)
    
    return FileResponse(audio_path, media_type="audio/wav")

@router.get("/calls/transcript/{log_id}")
async def get_transcript(log_id: str):
    """Generate and return transcript for a call"""
    try:
        # Debug: log the request
        print(f"Transcript requested for log_id: {log_id}")
        print(f"Current working directory: {os.getcwd()}")
        
        chatlog_path = find_chatlog(log_id)
        print(f"Found chatlog path: {chatlog_path}")
        
        if not chatlog_path:
            return JSONResponse({
                "error": "Chatlog not found",
                "log_id": log_id,
                "cwd": os.getcwd(),
                "searched_in": "data/chat"
            }, status_code=404)
        
        with open(chatlog_path, 'r') as f:
            chatlog = json.load(f)
        
        transcript = generate_transcript(chatlog)
        agent_name = extract_agent_name(chatlog_path)
        phone_number = extract_phone_number(chatlog_path)
        
        print(f"Generated transcript for {agent_name} - {phone_number}")
        
        return JSONResponse({
            "success": True,
            "transcript": transcript,
            "agent_name": agent_name,
            "phone_number": phone_number
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in get_transcript: {str(e)}")
        print(error_trace)
        return JSONResponse({
            "error": str(e),
            "trace": error_trace
        }, status_code=500)

@router.get("/calls/audio_transcript/{log_id}")
async def get_audio_transcript(log_id: str):
    """Generate transcript from audio file using Whisper"""
    try:
        print(f"Audio transcript requested for log_id: {log_id}")
        
        audio_path = Path(f"data/calls/{log_id}.wav")
        
        if not audio_path.exists():
            return JSONResponse({
                "error": "Audio file not found",
                "log_id": log_id
            }, status_code=404)
        
        # Get metadata from chatlog if available
        chatlog_path = find_chatlog(log_id)
        agent_name = "Unknown"
        phone_number = "Unknown"
        
        if chatlog_path:
            agent_name = extract_agent_name(chatlog_path) or "Unknown"
            phone_number = extract_phone_number(chatlog_path) or "Unknown"
        
        # Load Whisper model and transcribe
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")
        print(f"Transcribing audio file: {audio_path}")
        result = model.transcribe(str(audio_path))
        
        transcript_text = result["text"]
        
        return JSONResponse({
            "success": True,
            "transcript": transcript_text,
            "agent_name": agent_name,
            "phone_number": phone_number
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in get_audio_transcript: {str(e)}")
        print(error_trace)
        return JSONResponse({
            "error": str(e),
            "trace": error_trace
        }, status_code=500)

def find_chatlog(log_id: str):
    """Find chatlog file by log_id"""
    # Search in data/chat directory
    result = subprocess.run(
        ["find", "data/chat", "-name", f"*{log_id}*.json"],
        capture_output=True,
        text=True
    )
    
    print(f"Find command output: {result.stdout}")
    print(f"Find command stderr: {result.stderr}")
    
    files = result.stdout.strip().split('\n')
    # Filter for chatlog files (not context files)
    chatlog_files = [f for f in files if f and 'chatlog_' in f]
    
    print(f"Filtered chatlog files: {chatlog_files}")
    
    return chatlog_files[0] if chatlog_files else None

def extract_phone_number(chatlog_path: str):
    """Extract phone number from chatlog"""
    try:
        with open(chatlog_path, 'r') as f:
            chatlog = json.load(f)
        
        for message in chatlog.get('messages', []):
            if message.get('role') == 'assistant':
                content = message.get('content', [])
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            # Look for call command
                            if '"call"' in text:
                                try:
                                    # Parse the JSON command
                                    match = re.search(r'\[.*?\]', text, re.DOTALL)
                                    if match:
                                        commands = json.loads(match.group())
                                        for cmd in commands:
                                            if 'call' in cmd:
                                                return cmd['call'].get('destination')
                                except:
                                    pass
    except:
        pass
    
    return None

def extract_agent_name(chatlog_path: str):
    """Extract agent name from file path"""
    # Path format: data/chat/admin/AgentName/chatlog_xxx.json
    parts = Path(chatlog_path).parts
    if len(parts) >= 4:
        return parts[-2]  # Agent name is second to last
    return None

def generate_transcript(chatlog: dict):
    """Generate clean transcript from chatlog"""
    transcript_lines = []
    in_call = False
    
    for message in chatlog.get('messages', []):
        role = message.get('role')
        
        if role == 'assistant':
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        
                        # Check if call started
                        if '"call"' in text:
                            in_call = True
                            continue
                        
                        # Check if call ended
                        if '"hangup"' in text or '"end_call"' in text:
                            break
                        
                        # Extract speak commands
                        if in_call and '"speak"' in text:
                            try:
                                match = re.search(r'\[.*?\]', text, re.DOTALL)
                                if match:
                                    commands = json.loads(match.group())
                                    for cmd in commands:
                                        if 'speak' in cmd:
                                            speak_text = cmd['speak'].get('text', '')
                                            if speak_text:
                                                transcript_lines.append(f"AI: {speak_text}")
                            except:
                                pass
                        
                        # Extract DTMF commands
                        if in_call and '"send_dtmf"' in text:
                            try:
                                match = re.search(r'\[.*?\]', text, re.DOTALL)
                                if match:
                                    commands = json.loads(match.group())
                                    for cmd in commands:
                                        if 'send_dtmf' in cmd:
                                            digits = cmd['send_dtmf'].get('digits', '')
                                            if digits:
                                                transcript_lines.append(f"DTMF: {digits}")
                            except:
                                pass
        
        elif role == 'user' and in_call:
            # Extract user speech
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                # Skip system messages and commands
                if not content.startswith('[') and not content.startswith('{'):
                    transcript_lines.append(f"Human: {content.strip()}")
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '').strip()
                        if text and not text.startswith('[') and not text.startswith('{'):
                            transcript_lines.append(f"Human: {text}")
    
    return '\n\n'.join(transcript_lines)
