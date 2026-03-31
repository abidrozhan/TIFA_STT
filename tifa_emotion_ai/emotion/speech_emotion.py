"""
TIFA Emotion AI - Speech Emotion Recognition
=============================================
Emotion detection from audio using SpeechBrain wav2vec2.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import warnings

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class SpeechEmotionRecognizer:
    """
    Emotion recognition from audio using SpeechBrain.
    
    Uses wav2vec2-based model trained on IEMOCAP dataset.
    Detects emotions: neutral, happy, sad, angry, fear, surprise, disgust
    """
    
    # Emotion labels from IEMOCAP model
    # The model outputs: ang, hap, neu, sad (4 classes)
    # We map to our extended emotion set
    IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
    
    def __init__(self, model_source: str = None):
        """
        Initialize SpeechBrain emotion recognizer.
        
        Args:
            model_source: SpeechBrain model source path/name
        """
        self.model_source = model_source or config.EMOTION_MODEL
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._fallback_analyzer = None
        self._load_model()
    
    def _load_model(self):
        """Load SpeechBrain emotion classifier"""
        try:
            # Fix torchaudio compatibility with newer versions
            import torchaudio
            if not hasattr(torchaudio, 'list_audio_backends'):
                # For torchaudio 2.1+, create a dummy function
                torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
            
            from speechbrain.inference.classifiers import EncoderClassifier
            
            logger.info(f"Loading emotion model: {self.model_source}")
            
            # Suppress warnings during model load
            warnings.filterwarnings("ignore")
            
            self.classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts={"device": self.device}
            )
            
            logger.info("Emotion recognition model loaded successfully")
        
        except ImportError as e:
            logger.error("SpeechBrain not installed. Run: pip install speechbrain")
            self._init_fallback()
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            logger.info("Using SimpleEmotionAnalyzer as fallback")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback emotion analyzer"""
        self.classifier = None
        self._fallback_analyzer = SimpleEmotionAnalyzer()
    
    def predict(self, audio: np.ndarray, text: str = None) -> Tuple[str, float]:
        """
        Predict emotion from audio and/or text.
        
        Args:
            audio: Audio array (float32, mono, 16kHz)
            text: Transcribed text for keyword-based emotion detection
        
        Returns:
            Tuple of (emotion_label, confidence)
        """
        if self.classifier is None:
            # Use fallback analyzer if available
            if self._fallback_analyzer is not None:
                logger.info("Using fallback emotion analyzer")
                return self._fallback_analyzer.predict(audio=audio, text=text)
            logger.warning("Emotion model not loaded, returning neutral")
            return "neutral", 0.75
        
        if audio is None or len(audio) == 0:
            return "neutral", 0.0
        
        try:
            # Ensure audio is tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
            else:
                audio_tensor = audio
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Get prediction
            out_prob, score, index, label = self.classifier.classify_batch(
                audio_tensor.to(self.device)
            )
            
            # Extract results
            emotion = label[0].lower() if label else "neutral"
            confidence = float(score[0]) if score is not None else 0.5
            
            # Map short labels to full names
            emotion = self._map_emotion_label(emotion)
            
            logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            return emotion, confidence
        
        except Exception as e:
            logger.error(f"Emotion prediction error: {e}")
            # SpeechBrain model is broken, switch to fallback permanently
            logger.info("Switching to text-based emotion analyzer permanently")
            self._init_fallback()
            if self._fallback_analyzer and text:
                return self._fallback_analyzer.predict(text=text)
            return "neutral", 0.5
    
    def predict_batch(self, audio_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict emotions for multiple audio samples.
        
        Args:
            audio_list: List of audio arrays
        
        Returns:
            List of (emotion, confidence) tuples
        """
        results = []
        for audio in audio_list:
            results.append(self.predict(audio))
        return results
    
    def get_all_probabilities(self, audio: np.ndarray) -> dict:
        """
        Get probabilities for all emotion classes.
        
        Args:
            audio: Audio array
        
        Returns:
            Dict mapping emotion labels to probabilities
        """
        if self.classifier is None:
            return {e: 0.25 for e in self.IEMOCAP_LABELS}
        
        try:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
            else:
                audio_tensor = audio
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            out_prob, _, _, _ = self.classifier.classify_batch(
                audio_tensor.to(self.device)
            )
            
            # Convert to dict
            probs = out_prob[0].cpu().numpy()
            result = {}
            for i, label in enumerate(self.IEMOCAP_LABELS):
                if i < len(probs):
                    result[label] = float(probs[i])
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            return {e: 0.25 for e in self.IEMOCAP_LABELS}
    
    def _map_emotion_label(self, label: str) -> str:
        """Map model output labels to our emotion set"""
        label = label.lower().strip()
        
        # Handle IEMOCAP short labels
        label_map = {
            "ang": "angry",
            "hap": "happy", 
            "neu": "neutral",
            "sad": "sad",
            "fea": "fear",
            "sur": "surprise",
            "dis": "disgust",
            "anger": "angry",
            "happiness": "happy",
            "sadness": "sad",
            "fearful": "fear",
            "surprised": "surprise"
        }
        
        return label_map.get(label, label)
    
    @property
    def emotions(self) -> List[str]:
        """Get list of supported emotions"""
        return config.EMOTIONS


class SimpleEmotionAnalyzer:
    """
    Fallback emotion analyzer using audio features AND text keywords.
    Used when SpeechBrain model is not available.
    
    Analyzes:
    - Text keywords (senang, sedih, marah, dll)
    - Pitch (high = excited/happy, low = sad/angry)
    - Energy (high = angry/excited, low = sad/calm)
    """
    
    # Indonesian emotion keywords
    EMOTION_KEYWORDS = {
        'happy': [
            'senang', 'bahagia', 'gembira', 'suka', 'ceria', 'riang',
            'excited', 'happy', 'bagus', 'mantap', 'keren', 'wow',
            'hore', 'yeay', 'asyik', 'asik', 'menyenangkan', 'seru'
        ],
        'sad': [
            'sedih', 'kecewa', 'murung', 'galau', 'pilu', 'duka',
            'menyedihkan', 'nelangsa', 'terpuruk', 'patah hati',
            'menangis', 'nangis', 'sad', 'down'
        ],
        'angry': [
            'marah', 'kesal', 'jengkel', 'benci', 'emosi', 'geram',
            'sebal', 'dongkol', 'murka', 'berang', 'angry', 'sebel'
        ],
        'fear': [
            'takut', 'khawatir', 'cemas', 'was-was', 'ngeri',
            'panik', 'gelisah', 'takutnya', 'serem', 'horor'
        ],
        'surprise': [
            'kaget', 'terkejut', 'heran', 'wow', 'astaga',
            'ya ampun', 'waduh', 'lho', 'hah', 'apa'
        ]
    }
    
    def __init__(self):
        logger.info("Using enhanced SimpleEmotionAnalyzer with text keywords")
    
    def predict(self, audio: np.ndarray = None, sample_rate: int = 16000, text: str = None) -> Tuple[str, float]:
        """
        Predict emotion from audio features AND text keywords.
        
        Args:
            audio: Audio array (optional)
            sample_rate: Audio sample rate
            text: Transcribed text for keyword analysis
            
        Returns:
            Tuple of (emotion, confidence)
        """
        # First, try text-based detection (more reliable)
        if text:
            text_emotion, text_conf = self._predict_from_text(text)
            if text_conf > 0.6:
                return text_emotion, text_conf
        
        # Fallback to audio-based detection
        if audio is not None and len(audio) > 0:
            return self._predict_from_audio(audio, sample_rate)
        
        return "neutral", 0.5
    
    def _predict_from_text(self, text: str) -> Tuple[str, float]:
        """Predict emotion from text keywords"""
        text_lower = text.lower()
        
        emotion_scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            # Get emotion with highest score
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(0.9, 0.6 + (emotion_scores[best_emotion] * 0.1))
            logger.info(f"Text emotion detected: {best_emotion} (keywords found: {emotion_scores[best_emotion]})")
            return best_emotion, confidence
        
        return "neutral", 0.5
    
    def _predict_from_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Predict emotion from audio features"""
        try:
            import librosa
            
            # Get features
            rms = librosa.feature.rms(y=audio)[0]
            energy = np.mean(rms)
            
            # Zero crossing rate (speech rate indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            speech_rate = np.mean(zcr)
            
            # Pitch estimation
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
            pitch_vals = pitches[pitches > 0]
            avg_pitch = np.mean(pitch_vals) if len(pitch_vals) > 0 else 200
            
            # Rule-based classification
            if energy > 0.08 and speech_rate > 0.08:
                if avg_pitch > 220:
                    return "happy", 0.65
                else:
                    return "angry", 0.65
            elif energy < 0.03:
                if avg_pitch < 160:
                    return "sad", 0.6
                else:
                    return "neutral", 0.55
            else:
                return "neutral", 0.5
        
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return "neutral", 0.5

