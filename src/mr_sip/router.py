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

@router.get('/calls')
async def list_calls(raw: bool = False, exclude_numbers: str = '', sessions: bool = False):
    """List all call recordings with metadata"""
    # option: get raw data
    exclude_numbers = [num.strip() for num in exclude_numbers.split(',') if num.strip()]
    calls_dir = Path('data/calls')
    chicago_tz = pytz.timezone('America/Chicago')
    calls_dict = {}
    if calls_dir.exists():
        for wav_file in sorted(calls_dir.glob('*.wav'), key=os.path.getmtime, reverse=True):
            log_id = wav_file.stem
            chatlog_path = find_chatlog(log_id)
             
            phone_number = None
            agent_name = None
            session_log = None
            if chatlog_path:
                phone_number = extract_phone_number(chatlog_path)
                agent_name = extract_agent_name(chatlog_path)
                if sessions:
                    with open(chatlog_path, 'r') as f:
                        session_log = json.load(f)

            unique_key = f'{log_id}_{phone_number}'
            if unique_key in calls_dict:
                continue
            mtime_utc = datetime.fromtimestamp(wav_file.stat().st_mtime, tz=pytz.UTC)
            mtime_chicago = mtime_utc.astimezone(chicago_tz)
            mtime = mtime_chicago
            calls_dict[unique_key] = {'log_id': log_id, 'filename': wav_file.name, 'session_log': session_log if session_log else None, 'date': mtime.strftime('%m/%d'), 'time': mtime.strftime('%I:%M %p').lstrip('0'), 'agent_name': agent_name or 'Unknown', 'phone_number': phone_number or 'Unknown', 'session_path': f'/session/{agent_name}/{log_id}' if agent_name else None}
    calls = list(calls_dict.values())
    if exclude_numbers:
        calls = [
            call for call in calls 
            if not any(num in call['phone_number'] for num in exclude_numbers)
        ]

    if raw:
        return JSONResponse(calls)
    html = await render('calls', {'calls': calls})
    return HTMLResponse(html)

@router.get('/calls/audio/{log_id}')
async def get_audio(log_id: str):
    """Serve audio file for a call"""
    audio_path = Path(f'data/calls/{log_id}.wav')
    if not audio_path.exists():
        return JSONResponse({'error': 'Audio file not found'}, status_code=404)
    return FileResponse(audio_path, media_type='audio/wav')

@router.get('/calls/transcript/{log_id}')
async def get_transcript(log_id: str):
    """Generate and return transcript for a call"""
    try:
        chatlog_path = find_chatlog(log_id)
        if not chatlog_path:
            return JSONResponse({'error': 'Chatlog not found', 'log_id': log_id, 'cwd': os.getcwd(), 'searched_in': 'data/chat'}, status_code=404)
        with open(chatlog_path, 'r') as f:
            chatlog = json.load(f)
        transcript = generate_transcript(chatlog)
        agent_name = extract_agent_name(chatlog_path)
        phone_number = extract_phone_number(chatlog_path)
        return JSONResponse({'success': True, 'transcript': transcript, 'agent_name': agent_name, 'phone_number': phone_number})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse({'error': str(e), 'trace': error_trace}, status_code=500)

@router.get('/calls/audio_transcript/{log_id}')
async def get_audio_transcript(log_id: str):
    """Generate transcript from audio file using Whisper"""
    try:
        cache_dir = Path('data/calls/transcripts')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'{log_id}.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                return JSONResponse(cached_data)
        audio_path = Path(f'data/calls/{log_id}.wav')
        if not audio_path.exists():
            return JSONResponse({'error': 'Audio file not found', 'log_id': log_id}, status_code=404)
        chatlog_path = find_chatlog(log_id)
        agent_name = 'Unknown'
        phone_number = 'Unknown'
        if chatlog_path:
            agent_name = extract_agent_name(chatlog_path) or 'Unknown'
            phone_number = extract_phone_number(chatlog_path) or 'Unknown'
        model = whisper.load_model('base')
        result = model.transcribe(str(audio_path))
        transcript_text = result['text']
        response_data = {'success': True, 'transcript': transcript_text, 'agent_name': agent_name, 'phone_number': phone_number}
        with open(cache_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        return JSONResponse(response_data)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JSONResponse({'error': str(e), 'trace': error_trace}, status_code=500)

def find_chatlog(log_id: str):
    """Find chatlog file by log_id"""
    result = subprocess.run(['find', 'data/chat', '-name', f'*{log_id}*.json'], capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    chatlog_files = [f for f in files if f and 'chatlog_' in f]
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
                            if '"call"' in text:
                                try:
                                    match = re.search('\\[.*?\\]', text, re.DOTALL)
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
    parts = Path(chatlog_path).parts
    if len(parts) >= 4:
        return parts[-2]
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
                        if '"call"' in text:
                            in_call = True
                            continue
                        if '"hangup"' in text or '"end_call"' in text:
                            break
                        if in_call and '"speak"' in text:
                            try:
                                match = re.search('\\[.*?\\]', text, re.DOTALL)
                                if match:
                                    commands = json.loads(match.group())
                                    for cmd in commands:
                                        if 'speak' in cmd:
                                            speak_text = cmd['speak'].get('text', '')
                                            if speak_text:
                                                transcript_lines.append(f'AI: {speak_text}')
                            except:
                                pass
                        if in_call and '"send_dtmf"' in text:
                            try:
                                match = re.search('\\[.*?\\]', text, re.DOTALL)
                                if match:
                                    commands = json.loads(match.group())
                                    for cmd in commands:
                                        if 'send_dtmf' in cmd:
                                            digits = cmd['send_dtmf'].get('digits', '')
                                            if digits:
                                                transcript_lines.append(f'DTMF: {digits}')
                            except:
                                pass
        elif role == 'user' and in_call:
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                if not content.startswith('[') and (not content.startswith('{')):
                    transcript_lines.append(f'Human: {content.strip()}')
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '').strip()
                        if text and (not text.startswith('[')) and (not text.startswith('{')):
                            transcript_lines.append(f'Human: {text}')
    return '\n\n'.join(transcript_lines)
