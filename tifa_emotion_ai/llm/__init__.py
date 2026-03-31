"""
TIFA Emotion AI - LLM Module
============================
LLaMA integration via Ollama for response generation.
"""

from .ollama_client import OllamaClient
from .prompts import EmotionPromptBuilder
from .context import ConversationContext
from .knowledge_memory import KnowledgeMemory

__all__ = ["OllamaClient", "EmotionPromptBuilder", "ConversationContext", "KnowledgeMemory"]

