"""
TIFA Emotion AI - Emotion Module
================================
Emotion detection and classification components.
"""

from .speech_emotion import SpeechEmotionRecognizer
from .classifier import EmotionClassifier
from .dataset import EmotionDataset

__all__ = ["SpeechEmotionRecognizer", "EmotionClassifier", "EmotionDataset"]
