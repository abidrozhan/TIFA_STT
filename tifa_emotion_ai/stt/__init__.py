"""
TIFA Emotion AI - STT Module
============================
Speech-to-Text components using wav2vec2.
"""

from .wav2vec_stt import Wav2VecSTT
from .audio_processor import AudioProcessor

__all__ = ["Wav2VecSTT", "AudioProcessor"]
