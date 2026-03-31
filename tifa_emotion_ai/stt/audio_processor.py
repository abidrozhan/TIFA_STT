"""
TIFA Emotion AI - Audio Processor
=================================
Audio preprocessing for STT and emotion recognition.
Handles recording, resampling, and normalization.
"""

import numpy as np
import sounddevice as sd
import librosa
from typing import Optional, Tuple
import threading
import queue

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """
    Audio preprocessing pipeline for TIFA.
    
    Features:
    - Microphone recording with silence detection
    - Resampling to 16kHz
    - Mono conversion
    - Normalization
    """
    
    def __init__(
        self,
        sample_rate: int = None,
        silence_threshold: float = None,
        silence_duration: float = None
    ):
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.silence_threshold = silence_threshold or config.SILENCE_THRESHOLD
        self.silence_duration = silence_duration or config.SILENCE_DURATION
        
        # Recording state
        self._is_recording = False
        self._audio_queue = queue.Queue()
    
    def record_from_mic(
        self,
        max_duration: float = None,
        auto_stop_on_silence: bool = True
    ) -> Optional[np.ndarray]:
        """
        Record audio from microphone.
        
        Args:
            max_duration: Maximum recording duration in seconds
            auto_stop_on_silence: Stop recording after silence detected
        
        Returns:
            Audio array (float32, mono, 16kHz) or None if error
        """
        max_duration = max_duration or config.RECORD_DURATION
        
        try:
            # Get default input device sample rate
            device_info = sd.query_devices(kind='input')
            device_sr = int(device_info['default_samplerate'])
            
            logger.info(f"Recording from microphone (max {max_duration}s)...")
            
            audio_chunks = []
            silence_samples = 0
            silence_samples_threshold = int(self.silence_duration * device_sr)
            
            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                self._audio_queue.put(indata.copy())
            
            # Start recording
            with sd.InputStream(
                samplerate=device_sr,
                channels=1,
                dtype=np.float32,
                callback=callback,
                blocksize=int(device_sr * 0.1)  # 100ms blocks
            ):
                self._is_recording = True
                total_samples = 0
                max_samples = int(max_duration * device_sr)
                
                while self._is_recording and total_samples < max_samples:
                    try:
                        chunk = self._audio_queue.get(timeout=0.5)
                        audio_chunks.append(chunk)
                        total_samples += len(chunk)
                        
                        # Check for silence
                        if auto_stop_on_silence:
                            chunk_rms = np.sqrt(np.mean(chunk ** 2))
                            if chunk_rms < self.silence_threshold:
                                silence_samples += len(chunk)
                                if silence_samples >= silence_samples_threshold:
                                    logger.info("Silence detected, stopping recording")
                                    break
                            else:
                                silence_samples = 0
                    
                    except queue.Empty:
                        continue
                
                self._is_recording = False
            
            if not audio_chunks:
                logger.warning("No audio recorded")
                return None
            
            # Combine chunks
            audio = np.concatenate(audio_chunks, axis=0).flatten()
            
            # Resample if needed
            if device_sr != self.sample_rate:
                audio = self.resample(audio, device_sr, self.sample_rate)
            
            # Normalize
            audio = self.normalize(audio)
            
            logger.info(f"Recorded {len(audio) / self.sample_rate:.2f}s of audio")
            return audio
        
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
    
    def stop_recording(self):
        """Stop current recording"""
        self._is_recording = False
    
    def preprocess(
        self,
        audio: np.ndarray,
        source_rate: int = None
    ) -> np.ndarray:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio array
            source_rate: Source sample rate (if different from target)
        
        Returns:
            Preprocessed audio (float32, mono, 16kHz, normalized)
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if source_rate and source_rate != self.sample_rate:
            audio = self.resample(audio, source_rate, self.sample_rate)
        
        # Normalize
        audio = self.normalize(audio)
        
        return audio.astype(np.float32)
    
    def resample(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if source_rate == target_rate:
            return audio
        
        return librosa.resample(
            audio,
            orig_sr=source_rate,
            target_sr=target_rate
        )
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return audio, sr
    
    def save_audio_file(
        self,
        audio: np.ndarray,
        file_path: str,
        sample_rate: int = None
    ):
        """Save audio to file"""
        import soundfile as sf
        sample_rate = sample_rate or self.sample_rate
        sf.write(file_path, audio, sample_rate)
