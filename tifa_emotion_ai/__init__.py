"""
TIFA Emotion AI - Main Package
==============================
Emotion-aware conversational AI system for TIFA Robot.

Components:
- STT: wav2vec2-based Speech-to-Text
- Emotion: SpeechBrain emotion recognition + Trainable classifier
- LLM: Ollama/LLaMA response generation
- TTS: Coqui XTTSv2 with emotion modulation
"""

from .config import Config, config

__version__ = "2.0.0"
__author__ = "TIFA Team"

__all__ = [
    "Config",
    "config"
]
