"""
TIFA Emotion AI - STT Module
============================
Speech-to-Text using Faster-Whisper (offline).
"""

from .whisper_stt import WhisperSTT
from .audio_processor import AudioProcessor

__all__ = ["WhisperSTT", "AudioProcessor"]
