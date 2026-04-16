"""
TIFA Emotion AI - Configuration Management
==========================================
Centralized configuration for all components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import json
import os


@dataclass
class Config:
    """Main configuration class for TIFA Emotion AI System"""
    
    # ==================== Paths ====================
    BASE_DIR: Path = field(default_factory=lambda: Path("d:/Speech TT V2"))
    
    @property
    def PACKAGE_DIR(self) -> Path:
        return self.BASE_DIR / "tifa_emotion_ai"
    
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"
    
    @property
    def MODEL_DIR(self) -> Path:
        return self.DATA_DIR / "models"
    
    @property
    def EMOTION_SAMPLES_DIR(self) -> Path:
        return self.DATA_DIR / "emotion_samples"
    
    @property
    def TRAINING_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "training_data"
    
    # ==================== STT Settings ====================
    # Faster-Whisper for Indonesian speech recognition
    STT_MODEL_SIZE: str = "small"  # tiny/base/small/medium
    SAMPLE_RATE: int = 16000
    
    # ==================== Emotion Recognition ====================
    # Text-based emotion detection (keyword analysis)
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Supported emotions
    EMOTIONS: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"
    ])
    
    # Emotion label mapping (for display in Indonesian)
    EMOTION_LABELS_ID: Dict[str, str] = field(default_factory=lambda: {
        "neutral": "Netral",
        "happy": "Senang",
        "sad": "Sedih",
        "angry": "Marah",
        "fear": "Takut",
        "surprise": "Terkejut",
        "disgust": "Jijik"
    })
    
    # ==================== LLM Settings ====================
    OLLAMA_MODEL: str = "llama3.2:3b"  # 3b for better quality
    OLLAMA_HOST: str = "http://localhost:11434"
    MAX_CONTEXT_TURNS: int = 5
    RESPONSE_LANGUAGE: str = "Indonesian"
    
    # ==================== TTS Settings ====================
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    TTS_LANGUAGE: str = "en"  # Coqui doesn't support 'id', use 'en' (Edge TTS handles Indonesian)
    
    # Edge TTS voice for generating reference samples
    EDGE_TTS_VOICE: str = "id-ID-GadisNeural"
    
    # TTS emotion modulation parameters
    EMOTION_TTS_PARAMS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "neutral": {"rate": "+0%", "pitch": "+0Hz"},
        "happy": {"rate": "+15%", "pitch": "+10Hz"},
        "sad": {"rate": "-15%", "pitch": "-8Hz"},
        "angry": {"rate": "+10%", "pitch": "+5Hz"},
        "fear": {"rate": "+5%", "pitch": "+8Hz"},
        "surprise": {"rate": "+12%", "pitch": "+15Hz"},
        "disgust": {"rate": "-5%", "pitch": "-3Hz"}
    })
    
    # ==================== Audio Settings ====================
    RECORD_DURATION: float = 15.0  # Max recording duration in seconds
    SILENCE_THRESHOLD: float = 0.005 # Very sensitive for laptop mic
    SILENCE_DURATION: float = 2.0  # Stop recording after 2s of silence
    
    # ==================== Continuous Learning ====================
    AUTO_SAVE_INTERVAL: int = 10  # Save model every N interactions
    MIN_SAMPLES_FOR_TRAINING: int = 5
    
    def ensure_directories(self):
        """Create all required directories if they don't exist"""
        dirs = [
            self.DATA_DIR,
            self.MODEL_DIR,
            self.EMOTION_SAMPLES_DIR,
            self.TRAINING_DATA_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Path = None):
        """Save configuration to JSON file"""
        if path is None:
            path = self.BASE_DIR / "config.json"
        
        config_dict = {
            "STT_MODEL": self.STT_MODEL,
            "EMOTION_MODEL": self.EMOTION_MODEL,
            "OLLAMA_MODEL": self.OLLAMA_MODEL,
            "OLLAMA_HOST": self.OLLAMA_HOST,
            "CONFIDENCE_THRESHOLD": self.CONFIDENCE_THRESHOLD,
            "MAX_CONTEXT_TURNS": self.MAX_CONTEXT_TURNS,
            "SAMPLE_RATE": self.SAMPLE_RATE
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path = None) -> 'Config':
        """Load configuration from JSON file"""
        config = cls()
        if path is None:
            path = config.BASE_DIR / "config.json"
        
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        return config


# Global config instance
config = Config()
