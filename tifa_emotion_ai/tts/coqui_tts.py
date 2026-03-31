"""
TIFA Emotion AI - Coqui TTS Integration
=======================================
Text-to-Speech with emotion modulation using Coqui XTTSv2.
Includes fallback to Edge TTS for compatibility.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
import asyncio
import io

from ..config import config
from ..utils import get_logger
from .emotion_voice import EmotionVoiceManager

logger = get_logger(__name__)


class EmotionTTS:
    """
    Emotion-modulated Text-to-Speech.
    
    Primary: Coqui XTTSv2 with voice cloning from emotion reference
    Fallback: Edge TTS with pitch/rate modulation
    
    Features:
    - Emotion-based voice modulation
    - Multiple output formats
    - Streaming support (Edge TTS)
    """
    
    def __init__(
        self,
        use_coqui: bool = True,
        use_fallback: bool = True
    ):
        """
        Initialize TTS engine.
        
        Args:
            use_coqui: Try to use Coqui XTTSv2
            use_fallback: Use Edge TTS as fallback
        """
        self.use_coqui = use_coqui
        self.use_fallback = use_fallback
        
        self.voice_manager = EmotionVoiceManager()
        self.coqui_tts = None
        self.edge_tts_available = False
        
        # Initialize engines
        self._init_engines()
    
    def _init_engines(self):
        """Initialize TTS engines"""
        # Try Coqui TTS
        if self.use_coqui:
            try:
                from TTS.api import TTS
                
                logger.info("Loading Coqui XTTSv2 model...")
                self.coqui_tts = TTS(config.TTS_MODEL)
                
                # Move to GPU if available (optional, can be slow on CPU)
                # self.coqui_tts.to("cuda" if torch.cuda.is_available() else "cpu")
                
                logger.info("Coqui TTS loaded successfully")
            
            except ImportError:
                logger.warning("Coqui TTS not installed. Run: pip install TTS")
                self.coqui_tts = None
            except Exception as e:
                logger.warning(f"Coqui TTS failed to load: {e}")
                self.coqui_tts = None
        
        # Check Edge TTS availability
        try:
            import edge_tts
            self.edge_tts_available = True
            logger.info("Edge TTS available as fallback")
        except ImportError:
            self.edge_tts_available = False
            logger.warning("Edge TTS not available")
        
        # Ensure emotion samples exist for fallback
        if self.edge_tts_available and not self.voice_manager.has_all_samples():
            missing = self.voice_manager.get_missing_samples()
            logger.info(f"Missing emotion samples: {missing}")
    
    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        output_path: Optional[str] = None
    ) -> Optional[Union[np.ndarray, str]]:
        """
        Synthesize speech with emotion.
        
        Args:
            text: Text to synthesize
            emotion: Emotion for voice modulation
            output_path: If provided, save to file and return path
        
        Returns:
            Audio array (if no output_path) or path to saved file
        """
        if not text:
            return None
        
        emotion = emotion.lower()
        
        # Try Coqui first
        if self.coqui_tts is not None:
            result = self._synthesize_coqui(text, emotion, output_path)
            if result is not None:
                return result
        
        # Fallback to Edge TTS
        if self.edge_tts_available:
            return self._synthesize_edge(text, emotion, output_path)
        
        logger.error("No TTS engine available")
        return None
    
    def _synthesize_coqui(
        self,
        text: str,
        emotion: str,
        output_path: Optional[str] = None
    ) -> Optional[Union[np.ndarray, str]]:
        """Synthesize using Coqui XTTSv2"""
        try:
            # Get reference audio for emotion
            reference_path = self.voice_manager.get_reference_path(emotion)
            
            if not reference_path.exists():
                # Try to generate it
                logger.info(f"Generating reference sample for {emotion}")
                asyncio.run(self.voice_manager.generate_reference_sample(emotion))
            
            if not reference_path.exists():
                logger.warning(f"No reference audio for {emotion}, using neutral")
                reference_path = self.voice_manager.get_reference_path("neutral")
            
            # If still no reference, skip Coqui
            if not reference_path.exists():
                return None
            
            logger.info(f"Synthesizing with Coqui TTS (emotion: {emotion})")
            
            if output_path:
                # Save to file
                self.coqui_tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=str(reference_path),
                    language=config.TTS_LANGUAGE
                )
                return output_path
            else:
                # Return audio array
                audio = self.coqui_tts.tts(
                    text=text,
                    speaker_wav=str(reference_path),
                    language=config.TTS_LANGUAGE
                )
                return np.array(audio)
        
        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")
            return None
    
    def _synthesize_edge(
        self,
        text: str,
        emotion: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Synthesize using Edge TTS (fallback)"""
        try:
            import edge_tts
            
            # Get emotion parameters
            params = self.voice_manager.get_emotion_params(emotion)
            
            logger.info(f"Synthesizing with Edge TTS (emotion: {emotion})")
            
            async def _generate():
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=config.EDGE_TTS_VOICE,
                    rate=params["rate"],
                    pitch=params["pitch"]
                )
                
                if output_path:
                    await communicate.save(output_path)
                    return output_path
                else:
                    # Save to temp file
                    import tempfile
                    temp_path = tempfile.mktemp(suffix=".mp3")
                    await communicate.save(temp_path)
                    return temp_path
            
            # Run async function
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = loop.run_until_complete(_generate())
                else:
                    result = loop.run_until_complete(_generate())
            except RuntimeError:
                result = asyncio.run(_generate())
            
            return result
        
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return None
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        emotion: str = "neutral"
    ) -> bool:
        """
        Synthesize and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            emotion: Emotion for modulation
        
        Returns:
            True if successful
        """
        result = self.synthesize(text, emotion, output_path)
        return result is not None
    
    def get_available_emotions(self) -> list:
        """Get list of available emotions"""
        return list(self.voice_manager.EMOTION_CONFIGS.keys())
    
    def setup_references(self, overwrite: bool = False):
        """Generate all emotion reference samples"""
        self.voice_manager.generate_samples_sync(overwrite)
    
    @property
    def engine_name(self) -> str:
        """Get name of active TTS engine"""
        if self.coqui_tts is not None:
            return "Coqui XTTSv2"
        elif self.edge_tts_available:
            return "Edge TTS"
        else:
            return "None"


class SimpleTTS:
    """
    Simple TTS using Edge TTS only.
    Lighter weight alternative when Coqui is not needed.
    """
    
    def __init__(self):
        self.voice = config.EDGE_TTS_VOICE
        self.emotion_params = config.EMOTION_TTS_PARAMS
    
    async def synthesize_async(
        self,
        text: str,
        emotion: str = "neutral",
        output_path: str = None
    ) -> Optional[str]:
        """Async synthesis"""
        try:
            import edge_tts
            
            params = self.emotion_params.get(emotion, self.emotion_params["neutral"])
            
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=params["rate"],
                pitch=params["pitch"]
            )
            
            if output_path is None:
                import tempfile
                output_path = tempfile.mktemp(suffix=".mp3")
            
            await communicate.save(output_path)
            return output_path
        
        except Exception as e:
            logger.error(f"SimpleTTS error: {e}")
            return None
    
    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        output_path: str = None
    ) -> Optional[str]:
        """Sync synthesis wrapper"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self.synthesize_async(text, emotion, output_path)
                )
            else:
                return loop.run_until_complete(
                    self.synthesize_async(text, emotion, output_path)
                )
        except RuntimeError:
            return asyncio.run(self.synthesize_async(text, emotion, output_path))
