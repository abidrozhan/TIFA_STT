"""
TIFA Emotion AI - WebSocket Client
===================================
Sends TTS audio and emotion expressions to remote UI via WebSocket.
Audio is base64-encoded WAV sent through Cloudflare WebSocket.
"""

import json
import base64
import threading
import time
from typing import Optional
from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)


class TIFAWebSocketClient:
    """
    WebSocket client for sending TTS audio and expressions to UI.
    
    Protocol:
    1. Connect to wss://tifa-ws.forgixrobotic.com
    2. Send SI (Server Init) message
    3. Send EXPRESSION messages (sound + expression)
    """
    
    WS_URL = "wss://tifa-ws.forgixrobotic.com"
    UI_ID = "UI_TIFA_001"
    SERVER_ID = "SERVERAI001"
    
    # Map emotion to expression message
    EMOTION_TO_EXPRESSION = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "disgust": "disgust",
        "neutral": "neutral"
    }
    
    def __init__(self, ws_url: str = None):
        """
        Initialize WebSocket client.
        
        Args:
            ws_url: WebSocket URL (default: wss://tifa-ws.forgixrobotic.com)
        """
        self.ws_url = ws_url or self.WS_URL
        self.ws = None
        self.connected = False
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """
        Connect to WebSocket server and send SI init message.
        
        Returns:
            True if connected successfully
        """
        try:
            import websocket
            
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            
            self.ws = websocket.create_connection(
                self.ws_url,
                timeout=10,
                header={"User-Agent": "TIFA-EmotionAI/2.0"}
            )
            
            # Send initialization (SI) message
            si_message = {
                "code": "SI",
                "data": {
                    "type": "UI",
                    "ui_id": self.SERVER_ID
                }
            }
            self.ws.send(json.dumps(si_message))
            self.connected = True
            
            logger.info("WebSocket connected and initialized!")
            return True
            
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False
    
    def _ensure_connected(self) -> bool:
        """Ensure WebSocket is connected, reconnect if needed"""
        if self.connected and self.ws:
            try:
                self.ws.ping()
                return True
            except:
                self.connected = False
        
        return self.connect()
    
    def send_expression(self, emotion: str) -> bool:
        """
        Send expression/emotion to UI.
        
        Args:
            emotion: Emotion name (happy, sad, angry, etc)
            
        Returns:
            True if sent successfully
        """
        if not self._ensure_connected():
            return False
        
        expression = self.EMOTION_TO_EXPRESSION.get(emotion, "neutral")
        
        message = {
            "code": "EXPRESSION",
            "data": {
                "ui_id": self.UI_ID,
                "expression_type": "EXPRESSION",
                "message": expression
            }
        }
        
        return self._send(message)
    
    def send_audio_with_expression(
        self, 
        audio_path: str, 
        emotion: str
    ) -> bool:
        """
        Send WAV audio file as base64 with expression to UI.
        
        Args:
            audio_path: Path to WAV/MP3 audio file
            emotion: Emotion for expression
            
        Returns:
            True if sent successfully
        """
        if not self._ensure_connected():
            logger.warning("WebSocket not connected, skipping audio send")
            return False
        
        try:
            # Read audio file and encode to base64
            audio_file = Path(audio_path)
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            expression = self.EMOTION_TO_EXPRESSION.get(emotion, "neutral")
            
            # Send combined sound + expression
            message = {
                "code": "EXPRESSION",
                "data": {
                    "ui_id": self.UI_ID,
                    "expression_type": "SOUND_WITH_EXPRESSION",
                    "message": audio_base64,
                    "format": "wav",
                    "expression": expression
                }
            }
            
            success = self._send(message)
            if success:
                logger.info(f"Audio sent via WebSocket ({len(audio_bytes)} bytes, emotion: {emotion})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending audio via WebSocket: {e}")
            return False
    
    def send_audio_bytes(
        self, 
        audio_bytes: bytes, 
        emotion: str,
        audio_format: str = "wav"
    ) -> bool:
        """
        Send raw audio bytes as base64 with expression.
        
        Args:
            audio_bytes: Raw audio bytes
            emotion: Emotion for expression
            audio_format: Audio format (wav, pcm)
            
        Returns:
            True if sent successfully
        """
        if not self._ensure_connected():
            return False
        
        try:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            expression = self.EMOTION_TO_EXPRESSION.get(emotion, "neutral")
            
            message = {
                "code": "EXPRESSION",
                "data": {
                    "ui_id": self.UI_ID,
                    "expression_type": "SOUND_WITH_EXPRESSION",
                    "message": audio_base64,
                    "format": audio_format,
                    "expression": expression
                }
            }
            
            return self._send(message)
            
        except Exception as e:
            logger.error(f"Error sending audio bytes: {e}")
            return False
    
    def _send(self, message: dict) -> bool:
        """Send JSON message through WebSocket"""
        with self._lock:
            try:
                self.ws.send(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                self.connected = False
                return False
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None
            self.connected = False
            logger.info("WebSocket disconnected")
    
    def __del__(self):
        self.close()
