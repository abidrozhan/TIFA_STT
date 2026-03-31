"""
TIFA Emotion AI - Emotion Voice Manager
=======================================
Manages emotion reference audio samples for TTS voice modulation.
"""

import asyncio
from pathlib import Path
from typing import Dict, Optional
import os

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class EmotionVoiceManager:
    """
    Manages emotion reference audio samples.
    
    Uses Edge TTS to generate reference audio for each emotion,
    which can be used by Coqui XTTSv2 for voice cloning/style transfer.
    """
    
    # Voice characteristics for each emotion
    EMOTION_CONFIGS = {
        "neutral": {"rate": "+0%", "pitch": "+0Hz", "text": "Halo, ada yang bisa saya bantu hari ini?"},
        "happy": {"rate": "+15%", "pitch": "+10Hz", "text": "Wah senangnya! Aku sangat gembira bisa membantu!"},
        "sad": {"rate": "-15%", "pitch": "-8Hz", "text": "Aku turut prihatin mendengarnya..."},
        "angry": {"rate": "+10%", "pitch": "+5Hz", "text": "Aku mengerti perasaanmu, tenang dulu ya."},
        "fear": {"rate": "+5%", "pitch": "+8Hz", "text": "Tenang, jangan khawatir, semuanya akan baik-baik saja."},
        "surprise": {"rate": "+12%", "pitch": "+15Hz", "text": "Oh! Wah tidak disangka! Luar biasa!"},
        "disgust": {"rate": "-5%", "pitch": "-3Hz", "text": "Hmm, aku paham itu tidak nyaman."}
    }
    
    def __init__(self, samples_dir: Path = None):
        """
        Initialize voice manager.
        
        Args:
            samples_dir: Directory for reference samples
        """
        self.samples_dir = samples_dir or config.EMOTION_SAMPLES_DIR
        self.samples_dir = Path(self.samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.voice = config.EDGE_TTS_VOICE
    
    def get_reference_path(self, emotion: str) -> Path:
        """Get path to reference audio for emotion"""
        return self.samples_dir / f"reference_{emotion}.wav"
    
    def has_all_samples(self) -> bool:
        """Check if all emotion samples exist"""
        for emotion in config.EMOTIONS:
            if not self.get_reference_path(emotion).exists():
                return False
        return True
    
    def get_missing_samples(self) -> list:
        """Get list of missing emotion samples"""
        missing = []
        for emotion in config.EMOTIONS:
            if not self.get_reference_path(emotion).exists():
                missing.append(emotion)
        return missing
    
    async def generate_reference_sample(
        self,
        emotion: str,
        overwrite: bool = False
    ) -> Optional[Path]:
        """
        Generate reference audio sample for emotion using Edge TTS.
        
        Args:
            emotion: Emotion name
            overwrite: Whether to overwrite existing sample
        
        Returns:
            Path to generated sample or None
        """
        output_path = self.get_reference_path(emotion)
        
        if output_path.exists() and not overwrite:
            logger.info(f"Reference sample exists: {output_path}")
            return output_path
        
        if emotion not in self.EMOTION_CONFIGS:
            logger.warning(f"Unknown emotion: {emotion}")
            return None
        
        try:
            import edge_tts
            
            cfg = self.EMOTION_CONFIGS[emotion]
            
            logger.info(f"Generating reference sample for {emotion}...")
            
            communicate = edge_tts.Communicate(
                text=cfg["text"],
                voice=self.voice,
                rate=cfg["rate"],
                pitch=cfg["pitch"]
            )
            
            await communicate.save(str(output_path))
            
            logger.info(f"Generated: {output_path}")
            return output_path
        
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return None
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            return None
    
    async def generate_all_samples(self, overwrite: bool = False):
        """Generate reference samples for all emotions"""
        logger.info("Generating all emotion reference samples...")
        
        for emotion in config.EMOTIONS:
            await self.generate_reference_sample(emotion, overwrite)
        
        logger.info("All reference samples generated!")
    
    def generate_samples_sync(self, overwrite: bool = False):
        """Synchronous wrapper for generating samples"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(self.generate_all_samples(overwrite))
            else:
                loop.run_until_complete(self.generate_all_samples(overwrite))
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(self.generate_all_samples(overwrite))
    
    def get_emotion_params(self, emotion: str) -> Dict[str, str]:
        """Get TTS parameters for emotion"""
        return self.EMOTION_CONFIGS.get(emotion, self.EMOTION_CONFIGS["neutral"])
    
    def list_samples(self) -> Dict[str, bool]:
        """List all samples and their availability"""
        result = {}
        for emotion in config.EMOTIONS:
            result[emotion] = self.get_reference_path(emotion).exists()
        return result


# Convenience function for generating samples
def setup_emotion_samples(overwrite: bool = False):
    """Setup function to generate all emotion samples"""
    manager = EmotionVoiceManager()
    manager.generate_samples_sync(overwrite)
    return manager.list_samples()


if __name__ == "__main__":
    # Run sample generation
    print("Generating emotion reference samples...")
    results = setup_emotion_samples()
    print("\nSample availability:")
    for emotion, exists in results.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {emotion}")
