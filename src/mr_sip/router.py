from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from lib.templates import render
import os
import subprocess
from pathlib import Path
from datetime import datetime

router = APIRouter()

@router.get("/calls")
async def calls_page(request: Request):
    """Display list of call recordings."""
    user = request.state.user.username if hasattr(request.state, 'user') else 'guest'
    
    # Get list of .wav files from ./data/calls/
    calls_dir = Path("./data/calls")
    calls = []
    
    if calls_dir.exists():
        for wav_file in calls_dir.glob("*.wav"):
            log_id = wav_file.stem  # filename without extension
            
            # Find the agent name by searching for the chatlog file
            agent_name = None
            session_path = None
            
            try:
                # Search for chatlog file with this log_id
                result = subprocess.run(
                    ["find", ".", "-name", f"*{log_id}.json"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.stdout.strip():
                    # Parse path like ./admin/KatieRentalApplication/chatlog_F13tlVyjx4BZPpRoXeCWT.json
                    chatlog_path = result.stdout.strip().split('\n')[0]
                    parts = chatlog_path.split('/')
                    if len(parts) >= 3:
                        agent_name = parts[2]  # KatieRentalApplication
                        session_path = f"/session/{agent_name}/{log_id}"
            except Exception as e:
                print(f"Error finding agent for {log_id}: {e}")
            
            # Get file modification time
            mtime = datetime.fromtimestamp(wav_file.stat().st_mtime)
            
            calls.append({
                'log_id': log_id,
                'filename': wav_file.name,
                'agent_name': agent_name or 'Unknown',
                'session_path': session_path,
                'time': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': mtime.timestamp()
            })
    
    # Sort by timestamp, newest first
    calls.sort(key=lambda x: x['timestamp'], reverse=True)
    
    html = await render('calls', {
        "user": user,
        "calls": calls
    })
    return HTMLResponse(html)

@router.get("/calls/audio/{log_id}")
async def serve_audio(log_id: str):
    """Serve audio file for a specific call."""
    audio_path = Path(f"./data/calls/{log_id}.wav")
    
    if not audio_path.exists():
        return HTMLResponse("Audio file not found", status_code=404)
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{log_id}.wav"
    )
