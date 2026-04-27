"""
PySIP Process Wrapper - Runs PySIP in isolated process for smooth audio

This wrapper runs PySIP's SIP/RTP handling in a separate OS process to eliminate
GIL contention and ensure smooth audio processing. Communication happens via
multiprocessing queues.

Key benefits:
- No GIL contention between audio processing and main event loop
- True parallel execution on multi-core systems
- Isolated memory space prevents interference
- Minimal changes to existing code
"""
import multiprocessing as mp
import asyncio
import logging
import traceback
import time
from typing import Optional, Dict, Any
from datetime import datetime
logger = logging.getLogger(__name__)

class PySIPProcessWrapper:
    """Manages PySIP in a separate process with queue-based communication.
    
    This class runs in the main process and manages the PySIP subprocess.
    Audio flows through multiprocessing queues:
    - audio_in_queue: Phone audio -> Main process (for OpenAI)
    - audio_out_queue: Main process -> Phone audio (from OpenAI)
    - control_queue: Commands to PySIP process
    - status_queue: Status updates from PySIP process
    """

    def __init__(self, context=None):
        """Initialize the wrapper.
        
        Args:
            context: MindRoot context for logging
        """
        self.context = context
        self.audio_in_queue = mp.Queue(maxsize=200)
        self.audio_out_queue = mp.Queue(maxsize=200)
        self.control_queue = mp.Queue()
        self.status_queue = mp.Queue()
        self.process: Optional[mp.Process] = None
        self._running = False
        self._call_established = False
        self._audio_in_count = 0
        self._audio_out_count = 0
        self._start_time = None
        logger.info(f"PySIP process wrapper initialized for context {(context.log_id if context else 'N/A')}")

    async def start_call(self, user: str, password: str, gateway: str, destination: str, enable_recording: bool=False, recording_dir: str='recordings', record_separate: bool=False) -> bool:
        """Start PySIP in separate process and initiate call.
        
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway (host:port)
            destination: Phone number to call
            enable_recording: Enable call recording
            recording_dir: Directory for recordings
            record_separate: Save separate incoming/outgoing files
            
        Returns:
            True if call established successfully
            
        Raises:
            Exception if call fails to establish
        """
        logger.info(f'Starting PySIP process for call to {destination}')
        config = {'user': user, 'password': password, 'gateway': gateway, 'destination': destination, 'log_id': self.context.log_id if self.context else 'unknown', 'enable_recording': enable_recording, 'recording_dir': recording_dir, 'record_separate': record_separate}
        self.process = mp.Process(target=_run_pysip_process, args=(config, self.audio_in_queue, self.audio_out_queue, self.control_queue, self.status_queue), name=f"PySIP-{config['log_id']}")
        self.process.start()
        self._running = True
        self._start_time = time.time()
        logger.info(f'PySIP process started (PID: {self.process.pid})')
        try:
            status = await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.status_queue.get), timeout=120.0)
            if status['type'] == 'call_established':
                self._call_established = True
                logger.info(f'Call established to {destination} (took {time.time() - self._start_time:.2f}s)')
                return True
            elif status['type'] == 'call_failed':
                error = status.get('error', 'Unknown error')
                logger.error(f'Call failed: {error}')
                await self.stop()
                raise Exception(f'Call failed: {error}')
            else:
                logger.error(f'Unexpected status: {status}')
                await self.stop()
                raise Exception(f"Unexpected status: {status['type']}")
        except asyncio.TimeoutError:
            logger.error('Timeout waiting for call to establish')
            await self.stop()
            raise Exception('Call establishment timeout')
        finally:
            pass

    async def get_next_status(self) -> Optional[Dict[str, Any]]:
        """Get next status event from queue (non-blocking)."""
        if not self._running:
            return None
        else:
            pass
        try:
            return await asyncio.get_event_loop().run_in_executor(None, lambda: self.status_queue.get(block=False))
        except:
            return None
        finally:
            pass

    async def send_audio(self, audio_chunk: bytes, timestamp=None):
        """Send audio chunk to PySIP process (from OpenAI to phone).

        Args:
            audio_chunk: Audio data (ulaw 8kHz from OpenAI)
            timestamp:   Optional playback start timestamp from AudioPacer
        """
        if not self._running or not self._call_established:
            logger.warning('Cannot send audio - call not active')
            return
        else:
            pass
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self.audio_out_queue.put((audio_chunk, timestamp), block=True, timeout=0.1))
            self._audio_out_count += 1
            if self._audio_out_count % 50 == 0:
                logger.debug(f'Sent {self._audio_out_count} audio chunks to PySIP process')
            else:
                pass
        except Exception as e:
            logger.warning(f'Failed to queue audio chunk: {e}')
        finally:
            pass

    async def start_tts_response(self) -> bool:
        """Queue an ordered audio-response start marker to the PySIP subprocess."""
        if not self._running or not self._call_established:
            logger.warning('Cannot start audio response - call not active')
            return False
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self.audio_out_queue.put({'command': 'start_audio_response'}, block=True, timeout=0.1))
            return True
        except Exception as e:
            logger.warning(f'Failed to queue audio response start marker: {e}')
            return False

    async def end_tts_response(self) -> bool:
        """Queue an ordered audio-response end marker to the PySIP subprocess."""
        if not self._running or not self._call_established:
            logger.debug('Cannot end audio response - call not active')
            return False
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self.audio_out_queue.put({'command': 'end_audio_response'}, block=True, timeout=0.1))
            return True
        except Exception as e:
            logger.warning(f'Failed to queue audio response end marker: {e}')
            return False

    async def receive_audio(self) -> Optional[bytes]:
        """Receive audio chunk from PySIP process (from phone to OpenAI).
        
        Returns:
            Audio chunk or None if queue is empty
        """
        if not self._running:
            return None
        else:
            pass
        try:
            audio_chunk = await asyncio.get_event_loop().run_in_executor(None, lambda: self.audio_in_queue.get(block=False))
            self._audio_in_count += 1
            return audio_chunk
        except:
            return None
        finally:
            pass

    def clear_audio_queue(self):
        """Clear all queued outgoing audio (for interruption)."""
        try:
            cleared = 0
            while not self.audio_out_queue.empty():
                try:
                    self.audio_out_queue.get_nowait()
                    cleared += 1
                except:
                    break
                finally:
                    pass
            else:
                pass
            logger.info(f'Cleared {cleared} audio chunks from output queue')
            try:
                self.control_queue.put_nowait({'command': 'clear_audio'})
            except:
                pass
            finally:
                pass
        except Exception as e:
            logger.error(f'Error clearing audio queue: {e}')
        finally:
            pass

    def stop_silence_monitor(self):
        """Send command to subprocess to stop its silence monitor."""
        if not self._running:
            return
        else:
            pass
        try:
            self.control_queue.put_nowait({'command': 'stop_silence_monitor'})
            logger.info('Sent stop_silence_monitor command to subprocess')
        except Exception as e:
            logger.warning(f'Failed to send stop_silence_monitor command: {e}')
        finally:
            pass

    async def stop(self):
        """Stop the PySIP process and cleanup."""
        if not self._running:
            return
        else:
            pass
        logger.info('Stopping PySIP process...')
        self._running = False
        try:
            self.control_queue.put_nowait({'command': 'stop'})
        except:
            pass
        finally:
            pass
        if self.process and self.process.is_alive():
            try:
                await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.process.join, 5.0), timeout=6.0)
            except asyncio.TimeoutError:
                logger.warning('PySIP process did not exit cleanly, terminating...')
                self.process.terminate()
                await asyncio.sleep(1.0)
                if self.process.is_alive():
                    logger.error('PySIP process still alive, killing...')
                    self.process.kill()
                else:
                    pass
            finally:
                pass
        else:
            pass
        if self._start_time:
            duration = time.time() - self._start_time
            logger.info(f'PySIP process stopped. Duration: {duration:.1f}s, Audio in: {self._audio_in_count}, Audio out: {self._audio_out_count}')
        else:
            pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the PySIP process.
        
        Returns:
            Dictionary with metrics
        """
        return {'running': self._running, 'call_established': self._call_established, 'audio_in_count': self._audio_in_count, 'audio_out_count': self._audio_out_count, 'audio_in_queue_size': self.audio_in_queue.qsize() if self._running else 0, 'audio_out_queue_size': self.audio_out_queue.qsize() if self._running else 0, 'process_alive': self.process.is_alive() if self.process else False, 'uptime': time.time() - self._start_time if self._start_time else 0}

def _run_pysip_process(config: Dict[str, Any], audio_in_q: mp.Queue, audio_out_q: mp.Queue, control_q: mp.Queue, status_q: mp.Queue):
    """Main function for PySIP subprocess.
    
    This runs in a separate process and handles all PySIP operations.
    
    Args:
        config: Configuration dictionary
        audio_in_q: Queue for audio from phone to main process
        audio_out_q: Queue for audio from main process to phone
        control_q: Queue for control commands
        status_q: Queue for status updates
    """
    logging.basicConfig(level=logging.WARNING, format=f"[PySIP-{config['log_id']}] %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"PySIP subprocess started for {config['destination']}")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_pysip_main(config, audio_in_q, audio_out_q, control_q, status_q))
    except Exception as e:
        logger.error(f'Error in PySIP subprocess: {e}')
        logger.error(traceback.format_exc())
        try:
            status_q.put({'type': 'call_failed', 'error': str(e), 'timestamp': datetime.now().isoformat()})
        except:
            pass
        finally:
            pass
    finally:
        logger.info('PySIP subprocess exiting')

async def _pysip_main(config: Dict[str, Any], audio_in_q: mp.Queue, audio_out_q: mp.Queue, control_q: mp.Queue, status_q: mp.Queue):
    """Main async function for PySIP subprocess.
    
    Args:
        config: Configuration dictionary
        audio_in_q: Queue for audio from phone to main process
        audio_out_q: Queue for audio from main process to phone  
        control_q: Queue for control commands
        status_q: Queue for status updates
    """
    from .sip_client_s2s import MindRootSIPBotS2S
    logger = logging.getLogger(__name__)

    class MinimalContext:

        def __init__(self, log_id):
            self.log_id = log_id
    context = MinimalContext(config['log_id'])
    bot = MindRootSIPBotS2S(user=config['user'], password=config['password'], gateway=config['gateway'], context=context, enable_recording=config.get('enable_recording', False), recording_dir=config.get('recording_dir', 'recordings'), record_separate=config.get('record_separate', False), audio_in_queue=audio_in_q, audio_out_queue=audio_out_q, status_queue=status_q)
    audio_task = asyncio.create_task(_audio_queue_reader(bot, audio_out_q))
    control_task = asyncio.create_task(_control_queue_monitor(bot, control_q))
    try:
        call_task = asyncio.create_task(bot.make_call(config['destination']))
        logger.info('Waiting for call to be answered...')
        await bot.call_answered.wait()
        logger.info('Call answered, sending status')
        status_q.put({'type': 'call_established', 'timestamp': datetime.now().isoformat()})
        logger.info('Waiting for call to end...')
        await call_task
    except Exception as e:
        logger.error(f'Error in PySIP main: {e}')
        logger.error(traceback.format_exc())
        status_q.put({'type': 'call_failed', 'error': str(e), 'timestamp': datetime.now().isoformat()})
    finally:
        audio_task.cancel()
        control_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass
        finally:
            pass
        try:
            await control_task
        except asyncio.CancelledError:
            pass
        finally:
            pass
        logger.info('PySIP main exiting')

async def _audio_queue_reader(bot, audio_out_q: mp.Queue):
    """Read audio from queue and send to bot.
    

    Args:
        bot: MindRootSIPBotS2S instance
        audio_out_q: Queue with audio from main process
    """
    logger = logging.getLogger(__name__)
    logger.info('Audio queue reader started')
    try:
        while True:
            try:
                audio_item = await asyncio.get_event_loop().run_in_executor(None, lambda: audio_out_q.get(block=True, timeout=1.0))
                if audio_item is None:
                    break
                else:
                    pass
                if isinstance(audio_item, dict):
                    command = audio_item.get('command')
                    if command == 'start_audio_response':
                        await bot.start_tts_response()
                    elif command == 'end_audio_response':
                        await bot.end_tts_response()
                    else:
                        logger.warning(f'Unknown audio_out_queue command: {command}')
                    continue
                if isinstance(audio_item, tuple):
                    audio_chunk, timestamp = audio_item
                else:
                    audio_chunk, timestamp = (audio_item, None)
                await bot.send_tts_audio(audio_chunk, timestamp=timestamp)
            except Exception:
                await asyncio.sleep(0.01)
            finally:
                pass
        else:
            pass
    except asyncio.CancelledError:
        logger.info('Audio queue reader cancelled')
    finally:
        logger.info('Audio queue reader exiting')

async def _control_queue_monitor(bot, control_q: mp.Queue):
    """Monitor control queue for commands.
    
    Args:
        bot: MindRootSIPBotS2S instance
        control_q: Queue with control commands
    """
    logger = logging.getLogger(__name__)
    logger.info('Control queue monitor started')
    try:
        while True:
            try:
                cmd = await asyncio.get_event_loop().run_in_executor(None, lambda: control_q.get(block=True, timeout=1.0))
                if cmd['command'] == 'stop':
                    logger.info('Stop command received')
                    await bot.hangup_call()
                    break
                elif cmd['command'] == 'clear_audio':
                    logger.info('Clear audio command received')
                    bot.clear_audio_queue()
                elif cmd['command'] == 'stop_silence_monitor':
                    logger.info('Stop silence monitor command received')
                    bot.stop_silence_monitor()
                else:
                    pass
            except Exception:
                await asyncio.sleep(0.01)
            finally:
                pass
        else:
            pass
    except asyncio.CancelledError:
        logger.info('Control queue monitor cancelled')
    finally:
        logger.info('Control queue monitor exiting')