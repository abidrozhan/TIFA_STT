"""
TIFA Emotion AI - Ollama Client
===============================
LLaMA integration via Ollama API for response generation.
"""

import re
from typing import List, Dict, Optional, Generator
import requests

from ..config import config
from ..utils import get_logger
from .prompts import EmotionPromptBuilder
from .context import ConversationContext
from .knowledge_memory import KnowledgeMemory

logger = get_logger(__name__)


class OllamaClient:
    """
    Ollama client for LLaMA response generation.
    
    Features:
    - Emotion-aware response generation
    - Streaming support
    - Conversation context
    - Quick response patterns
    - Error handling and fallbacks
    """
    
    def __init__(
        self,
        model: str = None,
        host: str = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name (e.g., "llama3.2:3b")
            host: Ollama API host URL
        """
        self.model = model or config.OLLAMA_MODEL
        self.host = host or config.OLLAMA_HOST
        
        self.prompt_builder = EmotionPromptBuilder()
        self.context = ConversationContext()
        self.knowledge = KnowledgeMemory()
        
        # Check connection
        self._check_connection()
    
    def _check_connection(self, max_retries: int = 3) -> bool:
        """Check if Ollama is running with retry logic"""
        import time
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    
                    if self.model in model_names or any(self.model in n for n in model_names):
                        logger.info(f"Ollama connected, model '{self.model}' available")
                        return True
                    else:
                        logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
                        logger.info(f"Run: ollama pull {self.model}")
                        return False
                return False
            
            except requests.ConnectionError:
                if attempt < max_retries - 1:
                    logger.info(f"Ollama connection attempt {attempt + 1}/{max_retries} failed, retrying...")
                    time.sleep(1)
                else:
                    logger.warning("Ollama tidak berjalan. Jalankan: ollama serve")
                    return False
            except Exception as e:
                logger.error(f"Ollama connection error: {e}")
                return False
        
        return False
    
    def generate_response(
        self,
        user_text: str,
        emotion: str,
        confidence: float = 0.8,
        use_context: bool = True
    ) -> str:
        """
        Generate emotion-aware response.
        
        Args:
            user_text: User's transcribed message
            emotion: Detected emotion
            confidence: Emotion confidence
            use_context: Whether to include conversation history
        
        Returns:
            Generated response text
        """
        try:
            # Get relevant knowledge for this query
            knowledge_context = self.knowledge.get_knowledge_context(user_text)
            
            # Build messages with knowledge context
            history = self.context.get_messages() if use_context else None
            messages = self.prompt_builder.build_messages(
                user_text=user_text,
                emotion=emotion,
                confidence=confidence,
                history=history,
                knowledge_context=knowledge_context
            )
            
            # Call Ollama API
            response = self._call_ollama_chat(messages)
            
            if response:
                # Clean response
                response = self._clean_response(response)
                
                # Update context
                self._update_context(user_text, emotion, response)
                
                # Learn from this conversation
                self.knowledge.learn_from_conversation(user_text, response, emotion)
                
                return response
            else:
                return self._get_fallback_response(emotion)
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(emotion)
    
    def generate_stream(
        self,
        user_text: str,
        emotion: str,
        confidence: float = 0.8
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming.
        
        Args:
            user_text: User's message
            emotion: Detected emotion
            confidence: Confidence score
        
        Yields:
            Response chunks
        """
        try:
            history = self.context.get_messages()
            messages = self.prompt_builder.build_messages(
                user_text=user_text,
                emotion=emotion,
                confidence=confidence,
                history=history
            )
            
            full_response = ""
            
            for chunk in self._call_ollama_stream(messages):
                full_response += chunk
                yield chunk
            
            # Update context after complete
            self._update_context(user_text, emotion, full_response)
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield self._get_fallback_response(emotion)
    
    def _call_ollama_chat(self, messages: List[Dict]) -> Optional[str]:
        """Make chat API call to Ollama"""
        try:
            import ollama
            
            response = ollama.chat(
                model=self.model,
                messages=messages
            )
            
            return response.get("message", {}).get("content", "")
        
        except ImportError:
            # Fallback to REST API
            return self._call_ollama_rest(messages)
        
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return None
    
    def _call_ollama_rest(self, messages: List[Dict]) -> Optional[str]:
        """Fallback REST API call"""
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Ollama REST error: {e}")
            return None
    
    def _call_ollama_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Streaming API call"""
        try:
            import ollama
            
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
    
    def _check_quick_patterns(self, text: str, emotion: str) -> Optional[str]:
        """Check for quick response patterns"""
        text_lower = text.lower()
        
        # Greeting patterns
        greetings = ["halo", "hai", "hello", "hi", "selamat pagi", "selamat siang", 
                     "selamat sore", "selamat malam", "assalamualaikum", "hey"]
        if any(g in text_lower for g in greetings) and len(text_lower.split()) < 5:
            return get_quick_response("greeting", emotion)
        
        # Thanks patterns
        thanks = ["terima kasih", "makasih", "thanks", "thank you", "trims", "tengkyu"]
        if any(t in text_lower for t in thanks):
            return get_quick_response("thanks", emotion)
        
        # Goodbye patterns
        goodbye = ["sampai jumpa", "bye", "dah", "dadah", "selamat tinggal", 
                   "see you", "sampai nanti"]
        if any(g in text_lower for g in goodbye):
            return get_quick_response("goodbye", emotion)
        
        return None
    
    def _update_context(self, user_text: str, emotion: str, response: str):
        """Update conversation context"""
        self.context.add_user_message(user_text, emotion)
        self.context.add_assistant_message(response)
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        response = response.strip()
        
        # Remove quotes if entire response is quoted
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove any system prompt leakage
        if "kamu adalah" in response.lower() or "you are" in response.lower():
            lines = response.split('\n')
            response = '\n'.join(l for l in lines 
                                 if "kamu adalah" not in l.lower() 
                                 and "you are" not in l.lower())
        
        # Limit length (remove overly long responses)
        sentences = re.split(r'[.!?]', response)
        if len(sentences) > 4:
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def _get_fallback_response(self, emotion: str) -> str:
        """Get fallback response when generation fails"""
        fallbacks = {
            "neutral": "Maaf, aku sedang kesulitan memproses. Bisa kamu ulangi?",
            "happy": "Hmm, maaf ya! Bisa kamu ulangi dengan lebih jelas?",
            "sad": "Maaf ya, aku tidak mendengar dengan jelas. Coba lagi ya?",
            "angry": "Maaf, sepertinya ada masalah. Bisa diulangi?",
            "fear": "Tenang, aku di sini. Bisa kamu ulangi perkataanmu?",
            "surprise": "Oh! Maaf, aku tidak menangkap itu. Ulangi ya?",
            "disgust": "Maaf, aku tidak mendengar dengan baik. Ulangi please?"
        }
        return fallbacks.get(emotion, fallbacks["neutral"])
    
    def clear_context(self):
        """Clear conversation context"""
        self.context.clear()
    
    def new_session(self):
        """Start new conversation session"""
        self.context.start_new_session()
    
    def get_context_info(self) -> str:
        """Get context summary"""
        return self.context.get_context_summary()
