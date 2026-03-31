"""
TIFA Emotion AI - TTS Module
============================
Text-to-Speech with emotion modulation.
"""

from .coqui_tts import EmotionTTS
from .emotion_voice import EmotionVoiceManager

__all__ = ["EmotionTTS", "EmotionVoiceManager"]
