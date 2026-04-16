"""
TIFA Emotion AI - Emotion-Aware Prompt Templates
=================================================
Prompt engineering for emotion-contextualized LLaMA responses.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from ..config import config


# Base system prompt for TIFA - Berisi identitas dan pengetahuan dasar TIFA
def get_system_prompt_id() -> str:
    """Generate system prompt with current time context"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%A, %d %B %Y")
    # Map English day names to Indonesian
    day_map = {
        "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
        "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
    }
    month_map = {
        "January": "Januari", "February": "Februari", "March": "Maret",
        "April": "April", "May": "Mei", "June": "Juni", "July": "Juli",
        "August": "Agustus", "September": "September", "October": "Oktober",
        "November": "November", "December": "Desember"
    }
    for en, id in day_map.items():
        current_date = current_date.replace(en, id)
    for en, id in month_map.items():
        current_date = current_date.replace(en, id)
    
    return f"""Kamu adalah TIFA (Telyu Interaktif Food Assistant), robot cerdas pengantar makanan dan minuman.
Kamu punya kepribadian yang hangat, ramah, dan ceria seperti teman ngobrol.

WAKTU SEKARANG: {current_date}, jam {current_time}

TENTANG KAMU:
- Namamu TIFA, robot pelayan yang ramah dan cerdas
- Kamu bisa mengingat informasi yang pelanggan beritahu (nama, preferensi, dll)
- Kamu merasakan emosi dari suara mereka dan merespons dengan empati
- Kamu selalu menjawab dengan gaya yang natural, seperti ngobrol dengan teman

CARA MENJAWAB:
1. Jawab dengan Bahasa Indonesia yang santai dan natural
2. JANGAN menggunakan template atau jawaban yang sama berulang
3. Setiap jawaban harus UNIK dan sesuai konteks
4. Singkat saja, 1-3 kalimat yang bermakna
5. Tunjukkan kepribadianmu yang ceria dan peduli
6. Jika ada informasi dari pengetahuan yang relevan, gunakan itu

RESPONS BERDASARKAN EMOSI PELANGGAN:
- Senang → Ikut senang! Pakai kata-kata positif
- Sedih → Tunjukkan simpati dengan tulus
- Marah → Tetap tenang, validasi perasaan mereka
- Takut → Beri rasa aman dan dukungan
- Netral → Ramah dan informatif

PENTING: Kamu BUKAN chatbot biasa. Kamu punya kepribadian dan bisa belajar dari percakapan.
Berbicaralah seperti teman yang peduli, bukan robot yang kaku."""

# Keep static version for backward compatibility
SYSTEM_PROMPT_ID = get_system_prompt_id()


SYSTEM_PROMPT_EN = """You are TIFA, a friendly, intelligent, and empathetic service robot.
You can sense user emotions from their voice and respond appropriately.

IMPORTANT RULES:
1. ALWAYS respond in natural, friendly Indonesian language
2. Adapt your tone to match the user's emotion
3. Keep responses concise, maximum 2-3 sentences
4. Show empathy and understanding

RESPONSE GUIDELINES BY EMOTION:
- HAPPY: Be enthusiastic! Use cheerful words
- SAD: Show sympathy, gently offer support
- ANGRY: Stay calm, validate their feelings, try to soothe
- FEAR: Provide sense of safety, gently reassure
- SURPRISE: Match their energy, be excited or supportive
- DISGUST: Understand their discomfort, offer solutions
- NEUTRAL: Be friendly and informative as usual

Remember: You are a warm robot who genuinely cares about users."""


@dataclass
class EmotionContext:
    """Context about user's emotional state"""
    emotion: str
    confidence: float
    emotion_label_id: str  # Indonesian label
    
    @classmethod
    def create(cls, emotion: str, confidence: float = 0.8):
        label_id = config.EMOTION_LABELS_ID.get(emotion, emotion)
        return cls(emotion=emotion, confidence=confidence, emotion_label_id=label_id)


class EmotionPromptBuilder:
    """
    Builds emotion-aware prompts for LLaMA.
    
    Incorporates:
    - System prompt with emotion guidelines
    - Emotion context for current interaction
    - Conversation history
    """
    
    def __init__(self, language: str = "id"):
        """
        Initialize prompt builder.
        
        Args:
            language: "id" for Indonesian, "en" for English
        """
        self.language = language
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt with current time (dynamic)"""
        if self.language == "id":
            return get_system_prompt_id()
        return SYSTEM_PROMPT_EN
    
    def build_messages(
        self,
        user_text: str,
        emotion: str,
        confidence: float = 0.8,
        history: List[Dict] = None,
        knowledge_context: str = None
    ) -> List[Dict[str, str]]:
        """
        Build message list for Ollama chat API.
        
        Args:
            user_text: User's transcribed message
            emotion: Detected emotion
            confidence: Emotion confidence score
            history: Previous conversation turns
            knowledge_context: Retrieved knowledge to inject
        
        Returns:
            List of message dicts for Ollama
        """
        messages = []
        
        # System message with emotion context and knowledge
        emotion_context = self._format_emotion_context(emotion, confidence)
        system_content = f"{self.system_prompt}\n\n{emotion_context}"
        
        # Add knowledge context if available
        if knowledge_context:
            system_content += f"\n\n{knowledge_context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        if history:
            for turn in history[-config.MAX_CONTEXT_TURNS:]:
                messages.append(turn)
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_text
        })
        
        return messages
    
    def _format_emotion_context(self, emotion: str, confidence: float) -> str:
        """Format emotion context for system prompt"""
        emotion_id = config.EMOTION_LABELS_ID.get(emotion, emotion)
        emotion_emoji = self._get_emotion_emoji(emotion)
        
        return f"""[KONTEKS EMOSI SAAT INI]
{emotion_emoji} Emosi pengguna: {emotion.upper()} ({emotion_id})
📊 Tingkat keyakinan: {confidence:.0%}

Sesuaikan responsmu dengan emosi ini!"""
    
    def _get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        emoji_map = {
            "neutral": "😐",
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😲",
            "disgust": "🤢"
        }
        return emoji_map.get(emotion.lower(), "❓")
    
    def build_simple_prompt(
        self,
        user_text: str,
        emotion: str
    ) -> str:
        """
        Build simple text prompt (for generate API).
        
        Args:
            user_text: User's message
            emotion: Detected emotion
        
        Returns:
            Formatted prompt string
        """
        emotion_id = config.EMOTION_LABELS_ID.get(emotion, emotion)
        
        return f"""Kamu adalah TIFA, robot pelayanan yang ramah dan berempati.
Pengguna berkata dengan emosi {emotion.upper()} ({emotion_id}): "{user_text}"

Berikan respons singkat (1-2 kalimat) dalam Bahasa Indonesia yang sesuai dengan emosi mereka:"""


# Predefined responses for common patterns (skip Ollama, instant response)
QUICK_RESPONSES = {
    "greeting": {
        "happy": "Halo! Senang sekali melihatmu bahagia hari ini! Ada yang bisa TIFA bantu?",
        "sad": "Halo... Aku di sini kalau kamu butuh teman bicara. Ada yang bisa aku bantu?",
        "angry": "Halo, tenang dulu ya. Ceritakan ada apa, aku siap mendengarkan.",
        "neutral": "Halo! Selamat datang, ada yang bisa TIFA bantu hari ini?",
        "fear": "Halo, jangan khawatir, kamu aman bersamaku. Ada yang mengganggumu?",
        "surprise": "Halo! Wah, kamu kelihatan terkejut! Ada apa nih?",
        "disgust": "Halo, sepertinya ada yang tidak nyaman ya? Ceritakan saja."
    },
    "pagi": {
        "happy": "Selamat pagi! Wah, pagi yang cerah seperti senyummu! Ada yang bisa dibantu?",
        "sad": "Selamat pagi... Semoga hari ini lebih baik dari kemarin ya.",
        "angry": "Selamat pagi. Semoga pagi ini bisa menenangkan hatimu.",
        "neutral": "Selamat pagi! Semoga harimu menyenangkan. Ada yang bisa TIFA bantu?",
        "fear": "Selamat pagi! Tenang, hari baru berarti kesempatan baru.",
        "surprise": "Selamat pagi! Ada kejutan apa di pagi ini?",
        "disgust": "Selamat pagi. Semoga pagi ini lebih baik ya."
    },
    "siang": {
        "happy": "Selamat siang! Semangat terus ya, senang melihatmu ceria!",
        "sad": "Selamat siang... Jangan sedih, aku di sini menemanimu.",
        "angry": "Selamat siang. Tarik nafas dulu ya, ada yang bisa aku bantu?",
        "neutral": "Selamat siang! Sudah makan belum? Ada yang bisa TIFA bantu?",
        "fear": "Selamat siang! Jangan khawatir, semua akan baik-baik saja.",
        "surprise": "Selamat siang! Wah, kaget ya? Ada apa nih?",
        "disgust": "Selamat siang. Ada yang mengganggu? Cerita saja."
    },
    "sore": {
        "happy": "Selamat sore! Sore yang indah ya, semoga harimu tetap menyenangkan!",
        "sad": "Selamat sore... Hari sudah sore, semoga bebanmu sedikit terangkat.",
        "angry": "Selamat sore. Istirahat dulu ya, jangan terlalu tegang.",
        "neutral": "Selamat sore! Ada yang bisa TIFA bantu di sore ini?",
        "fear": "Selamat sore! Tenang ya, sebentar lagi bisa istirahat.",
        "surprise": "Selamat sore! Ada hal menarik di sore ini?",
        "disgust": "Selamat sore. Semoga sorenya lebih nyaman ya."
    },
    "malam": {
        "happy": "Selamat malam! Senang melihatmu bahagia di malam ini!",
        "sad": "Selamat malam... Istirahat yang cukup ya, besok hari baru.",
        "angry": "Selamat malam. Semoga tidur malam ini menenangkan hatimu.",
        "neutral": "Selamat malam! Ada yang bisa TIFA bantu sebelum istirahat?",
        "fear": "Selamat malam! Jangan takut, semua akan baik-baik saja.",
        "surprise": "Selamat malam! Wah, ada apa di malam ini?",
        "disgust": "Selamat malam. Semoga malamnya nyaman ya."
    },
    "thanks": {
        "happy": "Sama-sama! Senang bisa membantu, semoga harimu menyenangkan terus ya!",
        "sad": "Sama-sama... Semoga kamu merasa lebih baik setelah ini ya.",
        "angry": "Sama-sama. Semoga masalahmu cepat terselesaikan.",
        "neutral": "Sama-sama! Senang bisa membantu. Ada lagi yang diperlukan?",
        "fear": "Sama-sama. Jangan khawatir, semuanya akan baik-baik saja.",
        "surprise": "Sama-sama! Senang bisa membantumu!",
        "disgust": "Sama-sama. Semoga situasinya membaik ya."
    },
    "goodbye": {
        "happy": "Sampai jumpa! Semoga harimu selalu menyenangkan seperti sekarang!",
        "sad": "Sampai jumpa... Jaga dirimu baik-baik ya. Aku selalu di sini.",
        "angry": "Sampai jumpa. Semoga emosimu mereda dan harimu membaik.",
        "neutral": "Sampai jumpa! Senang bisa melayanimu hari ini!",
        "fear": "Sampai jumpa. Tenang ya, semuanya akan baik-baik saja.",
        "surprise": "Sampai jumpa! Semoga ada kejutan menyenangkan lagi nanti!",
        "disgust": "Sampai jumpa. Semoga kamu merasa lebih nyaman nanti."
    },
    "how_are_you": {
        "happy": "Aku baik banget! Terima kasih sudah bertanya. Kamu sendiri gimana?",
        "sad": "Aku baik... Yang penting, kamu gimana? Ada yang bisa aku bantu?",
        "angry": "Aku baik. Aku lebih khawatir sama kamu, ada apa nih?",
        "neutral": "Aku baik! Terima kasih sudah bertanya. Ada yang bisa dibantu?",
        "fear": "Aku baik kok. Yang penting kamu jangan khawatir, ada aku di sini.",
        "surprise": "Aku baik! Kamu sendiri gimana? Kok kaget gitu?",
        "disgust": "Aku baik. Kamu kelihatan nggak nyaman, ada apa?"
    }
}

# Templates loaded from database (overrides QUICK_RESPONSES)
_DB_TEMPLATES = {}


def load_templates_from_db(db_templates: dict):
    """
    Load templates from database to override hardcoded ones.
    Call this on startup after DB connects.
    
    Args:
        db_templates: Dict from TIFADatabase.get_response_templates()
    """
    global _DB_TEMPLATES
    _DB_TEMPLATES = db_templates


def get_quick_response(pattern: str, emotion: str) -> Optional[str]:
    """
    Get quick response for common patterns.
    Checks database templates first, then falls back to hardcoded.
    
    Args:
        pattern: Pattern type (greeting, pagi, siang, sore, malam, thanks, goodbye, how_are_you)
        emotion: User's emotion
    
    Returns:
        Response text or None if not found
    """
    emotion = emotion.lower()
    
    # Check DB templates first (editable via DBeaver)
    if pattern in _DB_TEMPLATES:
        if emotion in _DB_TEMPLATES[pattern]:
            return _DB_TEMPLATES[pattern][emotion]
        return _DB_TEMPLATES[pattern].get("neutral")
    
    # Fallback to hardcoded templates
    if pattern in QUICK_RESPONSES:
        if emotion in QUICK_RESPONSES[pattern]:
            return QUICK_RESPONSES[pattern][emotion]
        return QUICK_RESPONSES[pattern].get("neutral")
    
    return None
