"""
TIFA Emotion AI - wav2vec2 Speech-to-Text
=========================================
Speech recognition using wav2vec2 model from HuggingFace.
No Google/Cloud dependency - runs completely offline.
"""

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from typing import Optional, Tuple
import warnings

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class Wav2VecSTT:
    """
    Speech-to-Text using wav2vec2.
    
    Uses facebook/wav2vec2-large-xlsr-53 as base model.
    Supports multiple languages including Indonesian.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        """
        Initialize wav2vec2 STT.
        
        Args:
            model_name: HuggingFace model name/path
            device: torch device ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or config.STT_MODEL
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing wav2vec2 STT on {self.device}...")
        
        # Load processor and model
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load wav2vec2 model and processor"""
        try:
            # Suppress some warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            
            # Load model
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("wav2vec2 model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load wav2vec2 model: {e}")
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = None
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate (default: 16000)
        
        Returns:
            Transcribed text
        """
        sample_rate = sample_rate or config.SAMPLE_RATE
        
        if audio is None or len(audio) == 0:
            return ""
        
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Process audio
            input_values = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values
            
            # Move to device
            input_values = input_values.to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Clean up transcription
            transcription = self._clean_text(transcription)
            
            logger.info(f"Transcribed: '{transcription}'")
            return transcription
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def transcribe_batch(
        self,
        audio_list: list,
        sample_rate: int = None
    ) -> list:
        """
        Transcribe multiple audio samples.
        
        Args:
            audio_list: List of audio arrays
            sample_rate: Sample rate
        
        Returns:
            List of transcribed texts
        """
        sample_rate = sample_rate or config.SAMPLE_RATE
        
        if not audio_list:
            return []
        
        try:
            # Process all audio
            input_values = self.processor(
                audio_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).input_values
            
            input_values = input_values.to(self.device)
            
            # Batch inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode all
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(predicted_ids)
            
            return [self._clean_text(t) for t in transcriptions]
        
        except Exception as e:
            logger.error(f"Batch transcription error: {e}")
            return [""] * len(audio_list)
    
    def _clean_text(self, text: str) -> str:
        """Clean up transcription text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special tokens if any
        text = text.replace("[PAD]", "").replace("[UNK]", "")
        
        return text.strip()
    
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
            # Process
            input_values = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            max_probs, predicted_ids = torch.max(probs, dim=-1)
            
            # Average confidence (excluding padding)
            confidence = max_probs[max_probs > 0.1].mean().item()
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids)[0]
            transcription = self._clean_text(transcription)
            
            return transcription, confidence
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", 0.0


# Alternative: Use Whisper as fallback (if wav2vec2 fails)
class WhisperSTT:
    """
    Fallback STT using OpenAI Whisper (runs locally).
    Better accuracy but slower than wav2vec2.
    """
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded")
        except ImportError:
            logger.warning("Whisper not installed. Run: pip install openai-whisper")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self.model is None:
            return ""
        
        try:
            result = self.model.transcribe(
                audio,
                language="id",  # Indonesian
                fp16=False
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""
