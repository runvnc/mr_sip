#!/usr/bin/env python3
"""
Audio Combiner Module for MindRoot SIP

Combines two WAV files (encoded and decoded) into a single overlayed audio file.
Used for creating complete call recordings when calls end.

This version uses FFmpeg for more reliable audio processing.
"""

import os
import re
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class AudioCombiner:
    """Combines encoded and decoded WAV files into a single overlayed recording using FFmpeg."""
    
    def __init__(self):
        """Initialize the audio combiner."""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.debug("FFmpeg is available")
                return True
            else:
                logger.warning("FFmpeg check returned non-zero exit code")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"FFmpeg not found or not accessible: {e}")
            return False
    
    def find_matching_file(self, filepath: str) -> str:
        """
        Find the matching -enc.wav or -dec.wav file for the given file.
        
        Args:
            filepath: Path to either the -enc.wav or -dec.wav file
            
        Returns:
            Path to the matching file
            
        Raises:
            ValueError: If file doesn't match expected pattern
            FileNotFoundError: If matching file is not found
        """
        dir_path, filename = os.path.split(filepath)
        
        # Extract the base pattern (everything before -enc or -dec)
        match = re.match(r'(dump-.*)-(enc|dec)\.wav$', filename)
        if not match:
            raise ValueError(f"File {filename} doesn't match expected pattern dump-*-enc.wav or dump-*-dec.wav")
        
        base_pattern, side = match.groups()
        other_side = 'dec' if side == 'enc' else 'enc'
        
        # Look for the matching file
        expected_filename = f"{base_pattern}-{other_side}.wav"
        expected_path = os.path.join(dir_path, expected_filename)
        
        if os.path.exists(expected_path):
            return expected_path
        else:
            raise FileNotFoundError(f"Matching file not found: {expected_filename}")
    
    def _combine_conversations_internal(self, file1_path: str, file2_path: str, output_path: str) -> bool:
        """
        Internal method to combine two WAV files by overlaying them using FFmpeg.
        
        Args:
            file1_path: Path to first WAV file
            file2_path: Path to second WAV file  
            output_path: Path where combined file will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify input files exist
            if not os.path.exists(file1_path):
                logger.error(f"Input file not found: {file1_path}")
                return False
            if not os.path.exists(file2_path):
                logger.error(f"Input file not found: {file2_path}")
                return False
            
            # Check input file sizes
            size1 = os.path.getsize(file1_path)
            size2 = os.path.getsize(file2_path)
            logger.info(f"Input file sizes - File1: {size1:,} bytes, File2: {size2:,} bytes")
            
            if size1 <= 44:
                logger.error(f"File1 appears to be empty (only WAV header): {file1_path}")
                return False
            if size2 <= 44:
                logger.error(f"File2 appears to be empty (only WAV header): {file2_path}")
                return False
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Combining audio files using FFmpeg: {file1_path} and {file2_path}")
            
            # Use FFmpeg to combine the audio files
            # The amix filter mixes the two audio streams together
            # We normalize the inputs and use a duration of longest to ensure we get all audio
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', file1_path,
                '-i', file2_path,
                '-filter_complex',
                '[0:a][1:a]amix=inputs=2:duration=longest:normalize=0',
                '-ar', '8000',  # Sample rate for telephony
                '-ac', '1',     # Mono
                '-c:a', 'pcm_s16le',  # PCM 16-bit little-endian
                output_path
            ]
            
            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
            
            # Verify output file was created and has content
            if not os.path.exists(output_path):
                logger.error(f"Output file was not created: {output_path}")
                return False
            
            output_size = os.path.getsize(output_path)
            if output_size <= 44:
                logger.error(f"Output file appears to be empty (only WAV header): {output_path}")
                return False
            
            logger.info(f"Combined conversation saved to: {output_path}")
            logger.info(f"File sizes - Original 1: {size1:,} bytes, "
                       f"Original 2: {size2:,} bytes, "
                       f"Combined: {output_size:,} bytes")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg command timed out")
            return False
        except Exception as e:
            logger.error(f"Error combining audio files: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def combine_conversations(self, file1_path: str, file2_path: str, output_path: str) -> bool:
        """
        Combine two WAV files by overlaying them using FFmpeg.
        Will retry once after 1 second if the first attempt fails.
        
        Args:
            file1_path: Path to first WAV file
            file2_path: Path to second WAV file  
            output_path: Path where combined file will be saved
            
        Returns:
            True if successful, False otherwise
        """
        # First attempt
        success = self._combine_conversations_internal(file1_path, file2_path, output_path)
        
        if not success:
            logger.warning("First attempt to combine audio failed, waiting 1 second and retrying...")
            time.sleep(1.0)
            logger.info("Retrying audio combination...")
            success = self._combine_conversations_internal(file1_path, file2_path, output_path)
            
            if success:
                logger.info("Retry successful!")
            else:
                logger.error("Retry failed, giving up")
        
        return success
    
    def combine_call_recording(self, enc_file: str, dec_file: str, output_path: str) -> bool:
        """
        Combine a call's encoded and decoded WAV files into a single recording.
        
        Args:
            enc_file: Path to encoded WAV file
            dec_file: Path to decoded WAV file
            output_path: Path where combined file will be saved
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Combining call recording: {enc_file} + {dec_file} -> {output_path}")
        return self.combine_conversations(enc_file, dec_file, output_path)
    
    def combine_from_single_file(self, input_file: str, output_path: str) -> bool:
        """
        Combine audio files when given only one of the pair.
        Automatically finds the matching file and combines them.
        
        Args:
            input_file: Path to either -enc.wav or -dec.wav file
            output_path: Path where combined file will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            matching_file = self.find_matching_file(input_file)
            logger.info(f"Found matching file: {matching_file}")
            logger.info(f"Combining {input_file} with {matching_file}")
            return self.combine_conversations(input_file, matching_file, output_path)
            
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Error finding matching file: {e}")
            return False
    
    def get_output_filename(self, log_id: str, base_dir: str = "data/calls") -> str:
        """
        Generate output filename for combined call recording.
        
        Args:
            log_id: Call session/log ID
            base_dir: Base directory for call recordings
            
        Returns:
            Full path to output file
        """
        if not log_id:
            # Fallback to timestamp if no log_id available
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"call_recording_{timestamp}.wav"
        else:
            # Clean log_id to ensure valid filename
            clean_id = re.sub(r'[^\w\-_.]', '_', str(log_id))
            filename = f"{clean_id}.wav"
        
        return os.path.join(base_dir, filename)

# Convenience function for quick usage
def combine_call_audio(enc_file: str, dec_file: str, log_id: str, 
                      base_dir: str = "data/calls") -> Optional[str]:
    """
    Convenience function to combine call audio files.
    
    Args:
        enc_file: Path to encoded WAV file
        dec_file: Path to decoded WAV file
        log_id: Call session/log ID
        base_dir: Base directory for call recordings
        
    Returns:
        Path to combined file if successful, None otherwise
    """
    combiner = AudioCombiner()
    output_path = combiner.get_output_filename(log_id, base_dir)
    
    if combiner.combine_call_recording(enc_file, dec_file, output_path):
        return output_path
    else:
        return None
