"""
TIFA Emotion AI - Conversation Context Manager
===============================================
Manages conversation history and context for LLaMA.
"""

from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import json
from pathlib import Path

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class ConversationTurn:
    """Single turn in conversation"""
    
    def __init__(
        self,
        role: str,
        content: str,
        emotion: str = None,
        timestamp: str = None
    ):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.emotion = emotion
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to Ollama message format"""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def to_full_dict(self) -> Dict:
        """Convert to full dict with metadata"""
        return {
            "role": self.role,
            "content": self.content,
            "emotion": self.emotion,
            "timestamp": self.timestamp
        }


class ConversationContext:
    """
    Manages conversation context for LLaMA interactions.
    
    Features:
    - Rolling history buffer
    - Emotion tracking
    - Session management
    - Persistence
    """
    
    def __init__(
        self,
        max_turns: int = None,
        persist_dir: Path = None
    ):
        """
        Initialize context manager.
        
        Args:
            max_turns: Maximum conversation turns to remember
            persist_dir: Directory for persistence
        """
        self.max_turns = max_turns or config.MAX_CONTEXT_TURNS
        self.history: deque = deque(maxlen=self.max_turns * 2)  # *2 for user+assistant
        
        self.persist_dir = persist_dir or config.DATA_DIR
        self.persist_dir = Path(self.persist_dir)
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        self.total_turns = 0
        
        # Emotion history for this session
        self.emotion_history: List[str] = []
    
    def add_user_message(
        self,
        content: str,
        emotion: str = None
    ):
        """
        Add user message to context.
        
        Args:
            content: User's transcribed message
            emotion: Detected emotion
        """
        turn = ConversationTurn(
            role="user",
            content=content,
            emotion=emotion
        )
        self.history.append(turn)
        
        if emotion:
            self.emotion_history.append(emotion)
        
        self.total_turns += 1
    
    def add_assistant_message(self, content: str):
        """
        Add assistant (TIFA) response to context.
        
        Args:
            content: Generated response
        """
        turn = ConversationTurn(
            role="assistant",
            content=content
        )
        self.history.append(turn)
    
    def get_messages(
        self,
        last_n: int = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for Ollama.
        
        Args:
            last_n: Number of turns to get (None = all in buffer)
        
        Returns:
            List of message dicts
        """
        turns = list(self.history)
        
        if last_n:
            turns = turns[-(last_n * 2):]  # *2 for user+assistant pairs
        
        return [turn.to_dict() for turn in turns]
    
    def get_full_history(self) -> List[Dict]:
        """Get full history with metadata"""
        return [turn.to_full_dict() for turn in self.history]
    
    def get_dominant_emotion(self) -> Optional[str]:
        """
        Get most common emotion in this session.
        
        Returns:
            Dominant emotion or None
        """
        if not self.emotion_history:
            return None
        
        from collections import Counter
        counter = Counter(self.emotion_history)
        return counter.most_common(1)[0][0]
    
    def get_recent_emotion(self) -> Optional[str]:
        """Get most recent emotion"""
        return self.emotion_history[-1] if self.emotion_history else None
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        self.emotion_history.clear()
        self.total_turns = 0
        logger.info("Conversation context cleared")
    
    def start_new_session(self):
        """Start a new conversation session"""
        # Save current session if has content
        if self.history:
            self.save_session()
        
        # Reset
        self.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        logger.info(f"Started new session: {self.session_id}")
    
    def save_session(self):
        """Persist current session to disk"""
        if not self.history:
            return
        
        try:
            sessions_dir = self.persist_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            
            session_file = sessions_dir / f"session_{self.session_id}.json"
            
            data = {
                "session_id": self.session_id,
                "start_time": self.session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_turns": self.total_turns,
                "dominant_emotion": self.get_dominant_emotion(),
                "emotion_history": self.emotion_history,
                "conversation": self.get_full_history()
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session saved: {session_file}")
        
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def get_context_summary(self) -> str:
        """
        Get summary of current context for logging.
        
        Returns:
            Summary string
        """
        return (
            f"Session: {self.session_id} | "
            f"Turns: {self.total_turns} | "
            f"History: {len(self.history)} messages | "
            f"Dominant: {self.get_dominant_emotion() or 'N/A'}"
        )
    
    def __len__(self) -> int:
        return len(self.history)
