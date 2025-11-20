#!/usr/bin/env python3
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
        
        # Audio queues (bounded for low latency)
        # Smaller queues = lower latency but more risk of drops
        # 200 frames = ~4 seconds at 20ms per frame
        self.audio_in_queue = mp.Queue(maxsize=200)   # Phone -> OpenAI
        self.audio_out_queue = mp.Queue(maxsize=200)  # OpenAI -> Phone
        self.control_queue = mp.Queue()               # Commands
        self.status_queue = mp.Queue()                # Status updates
        
        self.process: Optional[mp.Process] = None
        self._running = False
        self._call_established = False
        
        # Metrics
        self._audio_in_count = 0
        self._audio_out_count = 0
        self._start_time = None
        
        logger.info(f"PySIP process wrapper initialized for context {context.log_id if context else 'N/A'}")
        
    async def start_call(self, user: str, password: str, gateway: str, 
                        destination: str, enable_recording: bool = False,
                        recording_dir: str = "recordings",
                        record_separate: bool = False) -> bool:
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
        logger.info(f"Starting PySIP process for call to {destination}")
        print(f"[WRAPPER] Starting PySIP process for call to {destination}")
        
        config = {
            'user': user,
            'password': password,
            'gateway': gateway,
            'destination': destination,
            'log_id': self.context.log_id if self.context else 'unknown',
            'enable_recording': enable_recording,
            'recording_dir': recording_dir,
            'record_separate': record_separate
        }
        
        print(f"[WRAPPER] Creating subprocess...")
        # Start the PySIP process
        self.process = mp.Process(
            target=_run_pysip_process,
            args=(config, self.audio_in_queue, self.audio_out_queue,
                  self.control_queue, self.status_queue),
            name=f"PySIP-{config['log_id']}"
        )
        self.process.start()
        self._running = True
        self._start_time = time.time()
        
        print(f"[WRAPPER] PySIP process started (PID: {self.process.pid}), waiting for status...")
        logger.info(f"PySIP process started (PID: {self.process.pid})")
        
        # Wait for call to be established (with timeout)
        try:
            status = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.status_queue.get
                ),
                timeout=120.0  # 2 minute timeout
            )
            print(f"[WRAPPER] Received status: {status}")
            
            if status['type'] == 'call_established':
                self._call_established = True
                logger.info(f"Call established to {destination} (took {time.time() - self._start_time:.2f}s)")
                return True
            elif status['type'] == 'call_failed':
                error = status.get('error', 'Unknown error')
                logger.error(f"Call failed: {error}")
                await self.stop()
                raise Exception(f"Call failed: {error}")
            else:
                logger.error(f"Unexpected status: {status}")
                await self.stop()
                raise Exception(f"Unexpected status: {status['type']}")
                
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for call to establish")
            await self.stop()
            raise Exception("Call establishment timeout")
            
    async def send_audio(self, audio_chunk: bytes, timestamp=None):
        """Send audio chunk to PySIP process (from OpenAI to phone).

        Args:
            audio_chunk: Audio data (ulaw 8kHz from OpenAI)
            timestamp:   Optional playback start timestamp from AudioPacer
        """
        if not self._running or not self._call_established:
            logger.warning("Cannot send audio - call not active")
            return

        try:
            # Non-blocking put with timeout; store (chunk, timestamp)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.audio_out_queue.put(
                    (audio_chunk, timestamp),
                    block=True,
                    timeout=0.1,
                ),
            )
            self._audio_out_count += 1

            # Periodic logging
            if self._audio_out_count % 50 == 0:
                logger.debug(
                    f"Sent {self._audio_out_count} audio chunks to PySIP process"
                )

        except Exception as e:
            logger.warning(f"Failed to queue audio chunk: {e}")
            
    async def receive_audio(self) -> Optional[bytes]:
        """Receive audio chunk from PySIP process (from phone to OpenAI).
        
        Returns:
            Audio chunk or None if queue is empty
        """
        if not self._running:
            return None
            
        try:
            # Non-blocking get
            audio_chunk = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.audio_in_queue.get(block=False)
            )
            self._audio_in_count += 1
            return audio_chunk
        except:
            return None
            
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
            logger.info(f"Cleared {cleared} audio chunks from output queue")
            
            # Send clear command to PySIP process
            try:
                self.control_queue.put_nowait({'command': 'clear_audio'})
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error clearing audio queue: {e}")
            
    async def stop(self):
        """Stop the PySIP process and cleanup."""
        if not self._running:
            return
            
        logger.info("Stopping PySIP process...")
        self._running = False
        
        # Send stop command
        try:
            self.control_queue.put_nowait({'command': 'stop'})
        except:
            pass
            
        # Wait for process to exit (with timeout)
        if self.process and self.process.is_alive():
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.process.join, 5.0
                    ),
                    timeout=6.0
                )
            except asyncio.TimeoutError:
                logger.warning("PySIP process did not exit cleanly, terminating...")
                self.process.terminate()
                await asyncio.sleep(1.0)
                if self.process.is_alive():
                    logger.error("PySIP process still alive, killing...")
                    self.process.kill()
                    
        # Log statistics
        if self._start_time:
            duration = time.time() - self._start_time
            logger.info(f"PySIP process stopped. Duration: {duration:.1f}s, "
                       f"Audio in: {self._audio_in_count}, Audio out: {self._audio_out_count}")
                       
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the PySIP process.
        
        Returns:
            Dictionary with metrics
        """
        return {
            'running': self._running,
            'call_established': self._call_established,
            'audio_in_count': self._audio_in_count,
            'audio_out_count': self._audio_out_count,
            'audio_in_queue_size': self.audio_in_queue.qsize() if self._running else 0,
            'audio_out_queue_size': self.audio_out_queue.qsize() if self._running else 0,
            'process_alive': self.process.is_alive() if self.process else False,
            'uptime': time.time() - self._start_time if self._start_time else 0
        }


def _run_pysip_process(config: Dict[str, Any], audio_in_q: mp.Queue, 
                       audio_out_q: mp.Queue, control_q: mp.Queue, 
                       status_q: mp.Queue):
    """Main function for PySIP subprocess.
    
    This runs in a separate process and handles all PySIP operations.
    
    Args:
        config: Configuration dictionary
        audio_in_q: Queue for audio from phone to main process
        audio_out_q: Queue for audio from main process to phone
        control_q: Queue for control commands
        status_q: Queue for status updates
    """
    # Set up logging for subprocess
    logging.basicConfig(
        level=logging.WARNING,
        format=f'[PySIP-{config["log_id"]}] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    print(f"[SUBPROCESS] PySIP subprocess started for {config['destination']}")
    
    logger.info(f"PySIP subprocess started for {config['destination']}")
    
    try:
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print(f"[SUBPROCESS] Running _pysip_main...")
        # Run the main PySIP logic
        loop.run_until_complete(
            _pysip_main(config, audio_in_q, audio_out_q, control_q, status_q)
        )
        
    except Exception as e:
        logger.error(f"Error in PySIP subprocess: {e}")
        logger.error(traceback.format_exc())
        
        # Send error status
        try:
            status_q.put({
                'type': 'call_failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        except:
            pass
            
    finally:
        logger.info("PySIP subprocess exiting")


async def _pysip_main(config: Dict[str, Any], audio_in_q: mp.Queue,
                     audio_out_q: mp.Queue, control_q: mp.Queue,
                     status_q: mp.Queue):
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
    
    print(f"[PYSIP_MAIN] Creating bot...")
    # Create a minimal context object for the bot
    class MinimalContext:
        def __init__(self, log_id):
            self.log_id = log_id
            
    context = MinimalContext(config['log_id'])
    
    print(f"[PYSIP_MAIN] Creating MindRootSIPBotS2S...")
    # Create bot with queue mode enabled
    bot = MindRootSIPBotS2S(
        user=config['user'],
        password=config['password'],
        gateway=config['gateway'],
        context=context,
        enable_recording=config.get('enable_recording', False),
        recording_dir=config.get('recording_dir', 'recordings'),
        record_separate=config.get('record_separate', False),
        # Enable queue mode
        audio_in_queue=audio_in_q,
        audio_out_queue=audio_out_q
    )
    
    print(f"[PYSIP_MAIN] Starting audio queue reader task...")
    # Start audio queue reader task
    audio_task = asyncio.create_task(
        _audio_queue_reader(bot, audio_out_q)
    )
    
    # Start control queue monitor
    control_task = asyncio.create_task(
        _control_queue_monitor(bot, control_q)
    )
    
    try:
        print(f"[PYSIP_MAIN] Calling bot.make_call({config['destination']})...")
        # Start the call as a task (don't block)
        call_task = asyncio.create_task(bot.make_call(config['destination']))
        
        print(f"[PYSIP_MAIN] Waiting for call_answered event...")
        # Wait for call to be answered (not ended!)
        logger.info("Waiting for call to be answered...")
        await bot.call_answered.wait()
        
        # Now send status - call is ready for audio
        logger.info("Call answered, sending status")
        status_q.put({
            'type': 'call_established',
            'timestamp': datetime.now().isoformat()
        })
        
        # Wait for call task to complete (call ends)
        logger.info("Waiting for call to end...")
        await call_task
        
    except Exception as e:
        logger.error(f"Error in PySIP main: {e}")
        logger.error(traceback.format_exc())
        
        # Send error status
        status_q.put({
            'type': 'call_failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        
    finally:
        # Cleanup
        audio_task.cancel()
        control_task.cancel()
        
        try:
            await audio_task
        except asyncio.CancelledError:
            pass
            
        try:
            await control_task
        except asyncio.CancelledError:
            pass
            
        logger.info("PySIP main exiting")


async def _audio_queue_reader(bot, audio_out_q: mp.Queue):
    """Read audio from queue and send to bot.
    

    Args:
        bot: MindRootSIPBotS2S instance
        audio_out_q: Queue with audio from main process
    """
    logger = logging.getLogger(__name__)
    logger.info("Audio queue reader started")

    try:
        while True:
            try:
                # Get audio item from queue (with timeout)
                audio_item = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: audio_out_q.get(block=True, timeout=1.0),
                )

                if audio_item is None:  # Sentinel to stop
                    break

                # Support both legacy (bytes) and new (bytes, timestamp) items
                if isinstance(audio_item, tuple):
                    audio_chunk, timestamp = audio_item
                else:
                    audio_chunk, timestamp = audio_item, None

                # Send to bot with timestamp
                await bot.send_tts_audio(audio_chunk, timestamp=timestamp)

            except Exception:
                # Timeout or other error - continue
                await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        logger.info("Audio queue reader cancelled")
    finally:
        logger.info("Audio queue reader exiting")


async def _control_queue_monitor(bot, control_q: mp.Queue):
    """Monitor control queue for commands.
    
    Args:
        bot: MindRootSIPBotS2S instance
        control_q: Queue with control commands
    """
    logger = logging.getLogger(__name__)
    logger.info("Control queue monitor started")
    
    try:
        while True:
            try:
                # Check for commands (non-blocking)
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: control_q.get(block=True, timeout=1.0)
                )
                
                if cmd['command'] == 'stop':
                    logger.info("Stop command received")
                    await bot.hangup_call()
                    break
                elif cmd['command'] == 'clear_audio':
                    logger.info("Clear audio command received")
                    bot.clear_audio_queue()
                    
            except Exception:
                # Timeout - continue
                await asyncio.sleep(0.01)
                
    except asyncio.CancelledError:
        logger.info("Control queue monitor cancelled")
    finally:
        logger.info("Control queue monitor exiting")
