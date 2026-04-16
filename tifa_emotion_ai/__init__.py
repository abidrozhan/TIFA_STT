"""
TIFA Emotion AI - Main Package
==============================
Emotion-aware conversational AI system for TIFA Robot.

Components:
- STT: Faster-Whisper Speech-to-Text (offline)
- Emotion: Text-based keyword emotion detection + Trainable classifier
- LLM: Ollama/LLaMA response generation
- TTS: Coqui XTTSv2 / Edge TTS with emotion modulation
- WebSocket: Send audio + expression to remote UI
"""

from .config import Config, config

__version__ = "2.1.0"
__author__ = "TIFA Team"

__all__ = [
    "Config",
    "config"
]

