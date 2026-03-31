"""
TIFA Emotion AI - Trainable Emotion Classifier
===============================================
Machine learning classifier with continuous/incremental learning capability.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings

from ..config import config
from ..utils import get_logger
from .dataset import EmotionDataset

logger = get_logger(__name__)


class EmotionClassifier:
    """
    Trainable emotion classifier with incremental learning.
    
    Combines:
    1. Text features (what the user says)
    2. Audio emotion (detected from voice)
    
    Uses SGDClassifier which supports partial_fit() for online learning,
    meaning it can learn continuously without full retraining.
    
    Features:
    - Incremental learning (partial_fit)
    - Automatic persistence
    - User feedback integration
    - Confidence scoring
    """
    
    def __init__(
        self,
        model_dir: Path = None,
        auto_load: bool = True
    ):
        """
        Initialize emotion classifier.
        
        Args:
            model_dir: Directory for model persistence
            auto_load: Whether to load existing model
        """
        self.model_dir = model_dir or config.MODEL_DIR
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / "emotion_classifier.joblib"
        self.encoder_path = self.model_dir / "emotion_encoder.joblib"
        
        # HashingVectorizer for text (memory efficient, no fitting needed)
        self.text_vectorizer = HashingVectorizer(
            n_features=2**12,  # 4096 features
            alternate_sign=False,
            ngram_range=(1, 2)
        )
        
        # Label encoder for emotions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.EMOTIONS)
        
        # SGDClassifier with log_loss for probability outputs
        self.classifier = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True,
            n_jobs=-1
        )
        
        # Track if model has been trained
        self._is_fitted = False
        self._sample_count = 0
        
        # Load existing model if available
        if auto_load:
            self.load()
    
    def _create_features(self, text: str, audio_emotion: str) -> np.ndarray:
        """
        Create feature vector from text and audio emotion.
        
        Args:
            text: Transcribed text
            audio_emotion: Emotion detected from audio
        
        Returns:
            Feature vector
        """
        # Text features
        text_features = self.text_vectorizer.transform([text]).toarray()[0]
        
        # Audio emotion as one-hot
        audio_onehot = np.zeros(len(config.EMOTIONS))
        if audio_emotion in config.EMOTIONS:
            idx = config.EMOTIONS.index(audio_emotion)
            audio_onehot[idx] = 1.0
        
        # Combine features
        features = np.concatenate([text_features, audio_onehot])
        
        return features
    
    def partial_train(
        self,
        text: str,
        audio_emotion: str,
        final_emotion: str
    ):
        """
        Incrementally train classifier with a single sample.
        
        This is the key method for continuous learning - it updates
        the model without needing to retrain from scratch.
        
        Args:
            text: User's transcribed text
            audio_emotion: Emotion detected from audio
            final_emotion: Target emotion label
        """
        try:
            # Create features
            features = self._create_features(text, audio_emotion)
            X = features.reshape(1, -1)
            
            # Encode label
            y = self.label_encoder.transform([final_emotion])
            
            # Partial fit (incremental learning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.classifier.partial_fit(
                    X, y, 
                    classes=np.arange(len(config.EMOTIONS))
                )
            
            self._is_fitted = True
            self._sample_count += 1
            
            # Auto-save periodically
            if self._sample_count % config.AUTO_SAVE_INTERVAL == 0:
                self.save()
                logger.info(f"Auto-saved model ({self._sample_count} samples)")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    def batch_train(
        self,
        texts: List[str],
        audio_emotions: List[str],
        final_emotions: List[str]
    ):
        """
        Train on a batch of samples.
        
        Args:
            texts: List of transcribed texts
            audio_emotions: List of audio emotions
            final_emotions: List of target emotions
        """
        if not texts:
            return
        
        try:
            # Create features for all samples
            X = np.array([
                self._create_features(text, audio_emo)
                for text, audio_emo in zip(texts, audio_emotions)
            ])
            
            # Encode labels
            y = self.label_encoder.transform(final_emotions)
            
            # Partial fit on batch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.classifier.partial_fit(
                    X, y,
                    classes=np.arange(len(config.EMOTIONS))
                )
            
            self._is_fitted = True
            self._sample_count += len(texts)
            
            logger.info(f"Batch trained on {len(texts)} samples")
        
        except Exception as e:
            logger.error(f"Batch training error: {e}")
    
    def predict(
        self,
        text: str,
        audio_emotion: str
    ) -> str:
        """
        Predict final emotion.
        
        Args:
            text: Transcribed text
            audio_emotion: Emotion from audio analysis
        
        Returns:
            Predicted emotion label
        """
        if not self._is_fitted:
            # If not trained yet, return audio emotion
            logger.warning("Classifier not trained, using audio emotion")
            return audio_emotion
        
        try:
            features = self._create_features(text, audio_emotion)
            X = features.reshape(1, -1)
            
            pred = self.classifier.predict(X)
            emotion = self.label_encoder.inverse_transform(pred)[0]
            
            return emotion
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return audio_emotion
    
    def predict_with_confidence(
        self,
        text: str,
        audio_emotion: str
    ) -> Tuple[str, float]:
        """
        Predict emotion with confidence score.
        
        Args:
            text: Transcribed text
            audio_emotion: Audio emotion
        
        Returns:
            Tuple of (emotion, confidence)
        """
        if not self._is_fitted:
            return audio_emotion, 0.5
        
        try:
            features = self._create_features(text, audio_emotion)
            X = features.reshape(1, -1)
            
            # Get probabilities
            probs = self.classifier.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            emotion = self.label_encoder.inverse_transform([pred_idx])[0]
            
            return emotion, float(confidence)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return audio_emotion, 0.5
    
    def get_all_probabilities(
        self,
        text: str,
        audio_emotion: str
    ) -> dict:
        """
        Get probabilities for all emotions.
        
        Args:
            text: Transcribed text
            audio_emotion: Audio emotion
        
        Returns:
            Dict mapping emotion to probability
        """
        if not self._is_fitted:
            # Return uniform distribution
            return {e: 1.0/len(config.EMOTIONS) for e in config.EMOTIONS}
        
        try:
            features = self._create_features(text, audio_emotion)
            X = features.reshape(1, -1)
            
            probs = self.classifier.predict_proba(X)[0]
            
            result = {}
            for i, emotion in enumerate(config.EMOTIONS):
                result[emotion] = float(probs[i]) if i < len(probs) else 0.0
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            return {e: 1.0/len(config.EMOTIONS) for e in config.EMOTIONS}
    
    def train_from_dataset(self, dataset: EmotionDataset):
        """
        Train from EmotionDataset.
        
        Args:
            dataset: EmotionDataset instance
        """
        if len(dataset) < config.MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Not enough samples ({len(dataset)}) for training")
            return
        
        texts, audio_emotions, final_emotions = dataset.export_for_training()
        self.batch_train(texts, audio_emotions, final_emotions)
    
    def save(self):
        """Save model to disk"""
        try:
            model_data = {
                "classifier": self.classifier,
                "is_fitted": self._is_fitted,
                "sample_count": self._sample_count
            }
            
            joblib.dump(model_data, self.model_path)
            joblib.dump(self.label_encoder, self.encoder_path)
            
            logger.info(f"Model saved to {self.model_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load(self) -> bool:
        """
        Load model from disk.
        
        Returns:
            True if model loaded successfully
        """
        if not self.model_path.exists():
            logger.info("No existing model found")
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            self.classifier = model_data["classifier"]
            self._is_fitted = model_data.get("is_fitted", True)
            self._sample_count = model_data.get("sample_count", 0)
            
            if self.encoder_path.exists():
                self.label_encoder = joblib.load(self.encoder_path)
            
            logger.info(f"Model loaded ({self._sample_count} samples trained)")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def reset(self):
        """Reset classifier to untrained state"""
        self.classifier = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        self._is_fitted = False
        self._sample_count = 0
        
        # Remove saved model
        if self.model_path.exists():
            self.model_path.unlink()
        
        logger.warning("Classifier reset to untrained state")
    
    @property
    def is_trained(self) -> bool:
        """Check if classifier has been trained"""
        return self._is_fitted
    
    @property
    def sample_count(self) -> int:
        """Get number of training samples"""
        return self._sample_count
