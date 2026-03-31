"""
TIFA Emotion AI - Knowledge Memory System
==========================================
Persistent knowledge storage with PostgreSQL for learning from conversations.
Enables TIFA to remember facts, preferences, and learn over time.
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..config import config
from ..utils import get_logger

logger = get_logger(__name__)


class KnowledgeMemory:
    """
    Knowledge memory system for TIFA.
    
    Stores and retrieves learned information from conversations:
    - User preferences (nama, menu favorit, dll)
    - Learned facts (informasi yang diajarkan)
    - Conversation patterns
    
    Uses PostgreSQL for persistent storage.
    """
    
    # Database connection config - Laptop B (PostgreSQL Server)
    DB_CONFIG = {
        'host': '172.20.10.2',  # Laptop B IP Address
        'port': 5432,
        'database': 'tifa_db',
        'user': 'postgres',
        'password': 'Roger@123#'
    }
    
    # Knowledge categories
    CATEGORIES = [
        'user_info',       # Nama, identitas pelanggan
        'preferences',     # Preferensi (menu favorit, dll)
        'learned_facts',   # Fakta yang diajarkan
        'menu_info',       # Info tentang menu
        'general'          # Informasi umum
    ]
    
    def __init__(self):
        """Initialize knowledge memory with PostgreSQL connection"""
        self.conn = None
        self._connect()
        self._init_tables()
    
    def _connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.DB_CONFIG)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL knowledge database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("Falling back to local JSON storage")
            self.conn = None
            return False
    
    def _init_tables(self):
        """Create knowledge tables if they don't exist"""
        if not self.conn:
            return
        
        try:
            with self.conn.cursor() as cur:
                # Main knowledge table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tifa_knowledge (
                        id SERIAL PRIMARY KEY,
                        category VARCHAR(50) NOT NULL,
                        key VARCHAR(255) NOT NULL,
                        value TEXT NOT NULL,
                        confidence FLOAT DEFAULT 1.0,
                        source VARCHAR(50) DEFAULT 'conversation',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INT DEFAULT 0,
                        UNIQUE(category, key)
                    )
                """)
                
                # Conversation log for learning
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tifa_conversations (
                        id SERIAL PRIMARY KEY,
                        user_input TEXT NOT NULL,
                        tifa_response TEXT NOT NULL,
                        emotion VARCHAR(20),
                        extracted_facts JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_knowledge_category 
                    ON tifa_knowledge(category)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_knowledge_key 
                    ON tifa_knowledge(key)
                """)
                
            logger.info("Knowledge tables initialized")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def save_fact(
        self, 
        category: str, 
        key: str, 
        value: str,
        confidence: float = 1.0,
        source: str = 'conversation'
    ) -> bool:
        """
        Save a fact to knowledge base.
        
        Args:
            category: Category (user_info, preferences, etc)
            key: Unique identifier for the fact
            value: The information to store
            confidence: How confident we are (0-1)
            source: Where this info came from
            
        Returns:
            True if saved successfully
        """
        if not self.conn:
            return self._save_to_json(category, key, value)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tifa_knowledge (category, key, value, confidence, source)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (category, key) 
                    DO UPDATE SET 
                        value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        updated_at = CURRENT_TIMESTAMP
                """, (category, key.lower(), value, confidence, source))
            
            logger.info(f"Saved knowledge: [{category}] {key} = {value[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error saving fact: {e}")
            return False
    
    def get_fact(self, category: str, key: str) -> Optional[str]:
        """
        Get a specific fact from knowledge base.
        
        Args:
            category: Category to search in
            key: Key to look up
            
        Returns:
            The value if found, None otherwise
        """
        if not self.conn:
            return self._get_from_json(category, key)
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT value FROM tifa_knowledge 
                    WHERE category = %s AND key = %s
                """, (category, key.lower()))
                
                row = cur.fetchone()
                if row:
                    # Update access count
                    cur.execute("""
                        UPDATE tifa_knowledge 
                        SET access_count = access_count + 1 
                        WHERE category = %s AND key = %s
                    """, (category, key.lower()))
                    return row['value']
            return None
        except Exception as e:
            logger.error(f"Error getting fact: {e}")
            return None
    
    def get_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Get relevant knowledge for a query using keyword matching.
        
        Args:
            query: The user's input to find relevant info for
            top_k: Maximum number of results
            
        Returns:
            List of relevant knowledge items
        """
        if not self.conn:
            return self._search_json(query, top_k)
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Search for matching knowledge
                placeholders = ' OR '.join(['key ILIKE %s OR value ILIKE %s'] * len(keywords))
                params = []
                for kw in keywords:
                    params.extend([f'%{kw}%', f'%{kw}%'])
                
                cur.execute(f"""
                    SELECT category, key, value, confidence 
                    FROM tifa_knowledge 
                    WHERE {placeholders}
                    ORDER BY confidence DESC, access_count DESC
                    LIMIT %s
                """, (*params, top_k))
                
                results = cur.fetchall()
                return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stopwords
        stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'adalah',
            'ini', 'itu', 'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita',
            'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana',
            'sudah', 'belum', 'akan', 'bisa', 'dapat', 'harus', 'mau',
            'tolong', 'mohon', 'ya', 'tidak', 'bukan', 'jangan'
        }
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords[:5]  # Limit to top 5 keywords
    
    def learn_from_conversation(
        self, 
        user_input: str, 
        tifa_response: str,
        emotion: str = None
    ) -> List[Dict]:
        """
        Extract and save facts from a conversation.
        
        Args:
            user_input: What the user said
            tifa_response: What TIFA replied
            emotion: Detected emotion
            
        Returns:
            List of extracted facts
        """
        extracted = []
        
        # Pattern matching for learning
        patterns = [
            # Nama pattern
            (r'nama\s+(?:saya|aku)\s+(?:adalah\s+)?(\w+)', 'user_info', 'nama_user'),
            (r'(?:saya|aku)\s+(?:adalah\s+)?(\w+)', 'user_info', 'nama_user'),
            (r'panggil\s+(?:saya|aku)\s+(\w+)', 'user_info', 'nama_user'),
            
            # Preferensi menu
            (r'(?:menu|makanan|minuman)\s+favorit\s+(?:saya|aku)\s+(?:adalah\s+)?(.+)', 'preferences', 'menu_favorit'),
            (r'(?:saya|aku)\s+suka\s+(.+)', 'preferences', 'kesukaan'),
            (r'(?:saya|aku)\s+tidak\s+suka\s+(.+)', 'preferences', 'tidak_suka'),
            
            # Teaching patterns
            (r'(?:ingat|catat|simpan)\s+(?:bahwa\s+)?(.+)', 'learned_facts', 'info'),
            (r'(?:tifa|kamu)\s+harus\s+(?:tau|tahu)\s+(?:bahwa\s+)?(.+)', 'learned_facts', 'info'),
        ]
        
        import re
        user_lower = user_input.lower()
        
        for pattern, category, key_base in patterns:
            match = re.search(pattern, user_lower)
            if match:
                value = match.group(1).strip()
                if value:
                    key = f"{key_base}_{datetime.now().strftime('%Y%m%d%H%M%S')}" if key_base == 'info' else key_base
                    self.save_fact(category, key, value)
                    extracted.append({
                        'category': category,
                        'key': key,
                        'value': value
                    })
        
        # Log conversation
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO tifa_conversations 
                        (user_input, tifa_response, emotion, extracted_facts)
                        VALUES (%s, %s, %s, %s)
                    """, (user_input, tifa_response, emotion, json.dumps(extracted)))
            except Exception as e:
                logger.error(f"Error logging conversation: {e}")
        
        if extracted:
            logger.info(f"Learned {len(extracted)} facts from conversation")
        
        return extracted
    
    def get_knowledge_context(self, query: str) -> str:
        """
        Get formatted knowledge context for injection into prompts.
        
        Args:
            query: User's query to find relevant knowledge
            
        Returns:
            Formatted string of relevant knowledge
        """
        relevant = self.get_relevant(query)
        
        if not relevant:
            return ""
        
        context_parts = ["[PENGETAHUAN YANG RELEVAN]"]
        for item in relevant:
            context_parts.append(f"- {item['key']}: {item['value']}")
        
        return "\n".join(context_parts)
    
    # === JSON Fallback Methods ===
    
    def _get_json_path(self) -> Path:
        return config.DATA_DIR / "knowledge.json"
    
    def _load_json(self) -> Dict:
        path = self._get_json_path()
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_json(self, data: Dict):
        path = self._get_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_to_json(self, category: str, key: str, value: str) -> bool:
        try:
            data = self._load_json()
            if category not in data:
                data[category] = {}
            data[category][key.lower()] = value
            self._save_json(data)
            return True
        except Exception as e:
            logger.error(f"JSON save error: {e}")
            return False
    
    def _get_from_json(self, category: str, key: str) -> Optional[str]:
        try:
            data = self._load_json()
            return data.get(category, {}).get(key.lower())
        except:
            return None
    
    def _search_json(self, query: str, top_k: int) -> List[Dict]:
        try:
            data = self._load_json()
            keywords = self._extract_keywords(query)
            results = []
            
            for category, items in data.items():
                for key, value in items.items():
                    for kw in keywords:
                        if kw in key.lower() or kw in value.lower():
                            results.append({
                                'category': category,
                                'key': key,
                                'value': value,
                                'confidence': 1.0
                            })
                            break
            
            return results[:top_k]
        except:
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")
