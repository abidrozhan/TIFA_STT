"""
TIFA Emotion AI - Emotion Dataset Manager
==========================================
Manages training data for continuous learning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import random

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionSample:
    """Single training sample for emotion classifier"""
    text: str
    audio_emotion: str
    final_emotion: str
    timestamp: str
    user_corrected: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EmotionSample':
        return cls(**data)


class EmotionDataset:
    """
    Manages training data for continuous learning emotion classifier.
    
    Features:
    - Add new samples during interaction
    - Persist to JSON file
    - Load historical data
    - Get training batches
    - Track user corrections for active learning
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory for storing training data
        """
        self.data_dir = data_dir or config.TRAINING_DATA_DIR
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.data_dir / "emotion_training_data.json"
        self.samples: List[EmotionSample] = []
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load existing training data from disk"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.samples = [
                        EmotionSample.from_dict(s) 
                        for s in data.get("samples", [])
                    ]
                logger.info(f"Loaded {len(self.samples)} training samples")
            except Exception as e:
                logger.error(f"Error loading training data: {e}")
                self.samples = []
        else:
            logger.info("No existing training data found")
            self.samples = []
    
    def save(self):
        """Persist training data to disk"""
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_samples": len(self.samples),
                "samples": [s.to_dict() for s in self.samples]
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.samples)} training samples")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def add_sample(
        self,
        text: str,
        audio_emotion: str,
        final_emotion: str,
        user_corrected: bool = False
    ):
        """
        Add new training sample.
        
        Args:
            text: Transcribed text from user
            audio_emotion: Emotion detected from audio
            final_emotion: Final classified emotion (may differ from audio)
            user_corrected: Whether user manually corrected the emotion
        """
        sample = EmotionSample(
            text=text,
            audio_emotion=audio_emotion,
            final_emotion=final_emotion,
            timestamp=datetime.now().isoformat(),
            user_corrected=user_corrected
        )
        
        self.samples.append(sample)
        logger.debug(f"Added sample: {text[:50]}... -> {final_emotion}")
    
    def add_correction(
        self,
        text: str,
        audio_emotion: str,
        corrected_emotion: str
    ):
        """
        Add user correction (higher weight in training).
        
        Args:
            text: Original text
            audio_emotion: Originally detected emotion
            corrected_emotion: User's corrected emotion label
        """
        self.add_sample(
            text=text,
            audio_emotion=audio_emotion,
            final_emotion=corrected_emotion,
            user_corrected=True
        )
    
    def get_training_batch(
        self,
        batch_size: int = 32,
        include_corrections: bool = True 
    ) -> List[EmotionSample]:
        """
        Get batch for training.
        
        Args:
            batch_size: Number of samples
            include_corrections: Whether to prioritize user corrections
        
        Returns:
            List of training samples
        """
        if not self.samples:
            return []
        
        if include_corrections:
            # Prioritize user corrections (2x weight)
            corrections = [s for s in self.samples if s.user_corrected]
            regular = [s for s in self.samples if not s.user_corrected]
            
            # Duplicate corrections for higher weight
            weighted_pool = corrections * 2 + regular
        else:
            weighted_pool = self.samples
        
        # Random sample
        batch_size = min(batch_size, len(weighted_pool))
        return random.sample(weighted_pool, batch_size)
    
    def get_all_samples(self) -> List[EmotionSample]:
        """Get all training samples"""
        return self.samples
    
    def get_samples_by_emotion(self, emotion: str) -> List[EmotionSample]:
        """Get samples for specific emotion"""
        return [s for s in self.samples if s.final_emotion == emotion]
    
    def get_recent_samples(self, n: int = 100) -> List[EmotionSample]:
        """Get most recent n samples"""
        return self.samples[-n:]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            "total_samples": len(self.samples),
            "user_corrections": sum(1 for s in self.samples if s.user_corrected),
            "emotion_distribution": {},
            "audio_emotion_distribution": {}
        }
        
        for emotion in config.EMOTIONS:
            stats["emotion_distribution"][emotion] = sum(
                1 for s in self.samples if s.final_emotion == emotion
            )
            stats["audio_emotion_distribution"][emotion] = sum(
                1 for s in self.samples if s.audio_emotion == emotion
            )
        
        return stats
    
    def clear(self):
        """Clear all training data"""
        self.samples = []
        if self.data_file.exists():
            self.data_file.unlink()
        logger.warning("Training data cleared")
    
    def export_for_training(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Export data for sklearn training.
        
        Returns:
            Tuple of (texts, audio_emotions, final_emotions)
        """
        texts = [s.text for s in self.samples]
        audio_emotions = [s.audio_emotion for s in self.samples]
        final_emotions = [s.final_emotion for s in self.samples]
        
        return texts, audio_emotions, final_emotions
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
