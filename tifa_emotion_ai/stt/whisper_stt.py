"""
TIFA Emotion AI - Faster Whisper Speech-to-Text
================================================
Speech recognition using faster-whisper (CTranslate2 optimized).
4x faster than OpenAI Whisper with same accuracy.
Runs completely offline - no internet needed.
"""

import numpy as np
from typing import Optional, Tuple
import warnings
import tempfile
import os

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class WhisperSTT:
    """
    Speech-to-Text using faster-whisper.
    
    Uses CTranslate2 optimized Whisper model for fast, accurate
    Indonesian speech recognition.
    
    Model sizes:
    - tiny:  ~75MB,  fastest,  lower accuracy
    - base:  ~150MB, fast,     good accuracy
    - small: ~500MB, moderate, great accuracy (recommended)
    - medium:~1.5GB, slower,   best accuracy
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize Whisper STT.
        
        Args:
            model_size: Model size (tiny/base/small/medium)
            device: Device to use (cpu/cuda)
            compute_type: Computation type (int8 for CPU, float16 for GPU)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        logger.info(f"Initializing Faster-Whisper STT ({model_size}) on {device}...")
        self._load_model()
    
    def _load_model(self):
        """Load faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper {self.model_size} model...")
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            logger.info("Faster-Whisper model loaded successfully")
        
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = None,
        language: str = "id"
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (float32, mono, 16kHz)
            sample_rate: Sample rate (default: 16000)
            language: Language code (default: 'id' for Indonesian)
        
        Returns:
            Transcribed text
        """
        sample_rate = sample_rate or config.SAMPLE_RATE
        
        if audio is None or len(audio) == 0:
            return ""
        
        if self.model is None:
            logger.error("Whisper model not loaded")
            return ""
        
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to [-1, 1]
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection - skip silence
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
            
            # Combine all segments
            text = " ".join([segment.text.strip() for segment in segments])
            text = self._clean_text(text)
            
            logger.info(f"Transcribed: '{text}' (lang: {info.language}, prob: {info.language_probability:.2f})")
            return text
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def get_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int = None
    ) -> Tuple[str, float]:
        """
        Get transcription with confidence score.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
        
        Returns:
            Tuple of (transcription, confidence)
        """
        sample_rate = sample_rate or config.SAMPLE_RATE
        
        if audio is None or len(audio) == 0:
            return "", 0.0
        
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            segments, info = self.model.transcribe(
                audio,
                language="id",
                beam_size=5,
                vad_filter=True
            )
            
            texts = []
            confidences = []
            for segment in segments:
                texts.append(segment.text.strip())
                confidences.append(segment.avg_logprob)
            
            text = " ".join(texts)
            text = self._clean_text(text)
            
            # Convert log probability to 0-1 confidence
            avg_conf = np.mean(confidences) if confidences else -1.0
            confidence = min(1.0, max(0.0, 1.0 + avg_conf))
            
            return text, confidence
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean up transcription text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip()
