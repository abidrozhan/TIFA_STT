"""
TIFA Emotion AI - Database Client
==================================
PostgreSQL database client via Cloudflare Tunnel.
Handles logging conversations, emotions, and WebSocket events.
Auto-starts cloudflared tunnel on connect.
"""

import psycopg2
from psycopg2 import pool
from typing import Optional, Dict, List
from datetime import datetime
import uuid
import subprocess
import time
import shutil
import os
import socket

from .utils import get_logger

logger = get_logger(__name__)


class TIFADatabase:
    """
    Database client for TIFA system.
    
    Connects to PostgreSQL via Cloudflare Tunnel (localhost:5002).
    Auto-starts cloudflared tunnel if not already running.
    """
    
    CLOUDFLARED_HOSTNAME = "postgres.forgixrobotic.com"
    TUNNEL_PORT = 5002
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5002,
        database: str = "tifa",
        user: str = "tifa",
        password: str = "TifaBot2025@"
    ):
        self.db_config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self.conn = None
        self.connected = False
        self.session_id = self._generate_session_id()
        self._tunnel_process = None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"session_{now}_{short_uuid}"
    
    def _is_port_open(self, port: int) -> bool:
        """Check if a port is already in use (tunnel running)"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _find_cloudflared(self) -> Optional[str]:
        """Find cloudflared executable path"""
        # Check PATH first
        path = shutil.which("cloudflared")
        if path:
            return path
        
        # Check common install locations on Windows
        common_paths = [
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
            r"C:\Program Files\cloudflared\cloudflared.exe",
            os.path.expanduser(r"~\cloudflared\cloudflared.exe"),
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
        
        return None
    
    def _start_tunnel(self) -> bool:
        """
        Auto-start cloudflared tunnel as background process.
        Skips if tunnel is already running.
        """
        # Check if tunnel is already running on port
        if self._is_port_open(self.TUNNEL_PORT):
            logger.info(f"Tunnel already running on port {self.TUNNEL_PORT}")
            return True
        
        # Find cloudflared
        cloudflared = self._find_cloudflared()
        if not cloudflared:
            logger.error("cloudflared not found! Install: winget install Cloudflare.cloudflared")
            return False
        
        try:
            logger.info(f"Starting cloudflared tunnel → {self.CLOUDFLARED_HOSTNAME}...")
            
            # Start cloudflared as background process (hidden window on Windows)
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE
            
            self._tunnel_process = subprocess.Popen(
                [cloudflared, "access", "tcp",
                 "--hostname", self.CLOUDFLARED_HOSTNAME,
                 "--url", f"localhost:{self.TUNNEL_PORT}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo
            )
            
            # Wait for tunnel to be ready (max 10 seconds)
            for i in range(20):
                time.sleep(0.5)
                if self._is_port_open(self.TUNNEL_PORT):
                    logger.info("Cloudflared tunnel started successfully!")
                    return True
            
            logger.error("Tunnel started but port not responding")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start tunnel: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Connect to PostgreSQL database.
        Auto-starts cloudflared tunnel if needed.
        """
        # Step 1: Ensure tunnel is running
        if not self._start_tunnel():
            logger.error("Cannot start cloudflared tunnel")
            return False
        
        # Step 2: Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(**self.db_config, connect_timeout=10)
            self.conn.autocommit = True
            self.connected = True
            logger.info(f"Database connected! Session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.connected = False
            return False
    
    def _ensure_connected(self) -> bool:
        """Ensure database connection is alive"""
        if not self.connected or not self.conn:
            return self.connect()
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except:
            self.connected = False
            return self.connect()
    
    # ==================== CONVERSATION ====================
    
    def log_conversation(
        self,
        user_input: str,
        tifa_response: str,
        emotion_code: str,
        emotion_confidence: float,
        response_source: str = "ollama",
        response_time_ms: int = None
    ) -> bool:
        """
        Log a conversation to h_conversation.
        
        Args:
            user_input: What the user said
            tifa_response: TIFA's response
            emotion_code: Detected emotion
            emotion_confidence: Emotion confidence score
            response_source: 'template' or 'ollama'
            response_time_ms: Response generation time in ms
        """
        if not self._ensure_connected():
            return False
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO h_conversation 
                (session_id, user_input, tifa_response, emotion_code, 
                 emotion_confidence, response_source, response_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, user_input, tifa_response,
                emotion_code, emotion_confidence, response_source,
                response_time_ms
            ))
            cur.close()
            logger.info("Conversation logged to database")
            return True
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
            return False
    
    # ==================== EMOTION LOG ====================
    
    def log_emotion(
        self,
        emotion_code: str,
        confidence: float,
        detection_method: str = "text",
        user_text: str = None
    ) -> bool:
        """
        Log emotion detection to h_emotion_log.
        
        Args:
            emotion_code: Detected emotion
            confidence: Confidence score
            detection_method: 'text' or 'audio'
            user_text: Text that triggered emotion
        """
        if not self._ensure_connected():
            return False
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO h_emotion_log 
                (session_id, emotion_code, confidence, detection_method, user_text)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                self.session_id, emotion_code, confidence,
                detection_method, user_text
            ))
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log emotion: {e}")
            return False
    
    # ==================== WEBSOCKET LOG ====================
    
    def log_websocket(
        self,
        message_type: str,
        expression: str = None,
        audio_size_bytes: int = None,
        status: str = "success",
        error_message: str = None
    ) -> bool:
        """
        Log WebSocket event to h_websocket_log.
        
        Args:
            message_type: SI, EXPRESSION, SOUND_WITH_EXPRESSION
            expression: Emotion expression sent
            audio_size_bytes: Size of audio sent
            status: 'success' or 'failed'
            error_message: Error message if failed
        """
        if not self._ensure_connected():
            return False
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO h_websocket_log 
                (session_id, message_type, expression, audio_size_bytes, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, message_type, expression,
                audio_size_bytes, status, error_message
            ))
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log websocket: {e}")
            return False
    
    # ==================== KNOWLEDGE ====================
    
    def save_knowledge(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "conversation"
    ) -> bool:
        """Save or update knowledge in g_knowledge"""
        if not self._ensure_connected():
            return False
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO g_knowledge (category, key, value, confidence, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (category, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
            """, (category, key, value, confidence, source))
            cur.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
            return False
    
    def get_knowledge(self, category: str = None) -> List[Dict]:
        """Get knowledge facts from g_knowledge"""
        if not self._ensure_connected():
            return []
        
        try:
            cur = self.conn.cursor()
            if category:
                cur.execute(
                    "SELECT category, key, value, confidence FROM g_knowledge WHERE category = %s",
                    (category,)
                )
            else:
                cur.execute("SELECT category, key, value, confidence FROM g_knowledge")
            
            rows = cur.fetchall()
            cur.close()
            return [
                {"category": r[0], "key": r[1], "value": r[2], "confidence": r[3]}
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            return []
    
    # ==================== RESPONSE TEMPLATES ====================
    
    def get_response_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load response templates from m_response_template.
        
        Returns:
            Dict like: {"pagi": {"happy": "Selamat pagi!...", "sad": "..."}, ...}
        """
        if not self._ensure_connected():
            return {}
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT pattern_code, emotion_code, response_text
                FROM m_response_template
                WHERE is_active = TRUE
                ORDER BY pattern_code, emotion_code
            """)
            rows = cur.fetchall()
            cur.close()
            
            templates = {}
            for pattern_code, emotion_code, response_text in rows:
                if pattern_code not in templates:
                    templates[pattern_code] = {}
                templates[pattern_code][emotion_code] = response_text
            
            logger.info(f"Loaded {len(rows)} response templates from database")
            return templates
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return {}
    
    def seed_templates(self, templates: dict) -> bool:
        """
        Seed response templates to m_response_template from Python dict.
        Only inserts if not already exists.
        
        Args:
            templates: Dict like {"pagi": {"happy": "text", ...}, ...}
        """
        if not self._ensure_connected():
            return False
        
        try:
            cur = self.conn.cursor()
            count = 0
            for pattern_code, emotions in templates.items():
                for emotion_code, response_text in emotions.items():
                    cur.execute("""
                        INSERT INTO m_response_template (pattern_code, emotion_code, response_text)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (pattern_code, emotion_code) DO NOTHING
                    """, (pattern_code, emotion_code, response_text))
                    count += 1
            cur.close()
            logger.info(f"Seeded {count} response templates")
            return True
        except Exception as e:
            logger.error(f"Failed to seed templates: {e}")
            return False
    
    # ==================== CLEANUP ====================
    
    def close(self):
        """Close database connection and tunnel"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None
            self.connected = False
            logger.info("Database disconnected")
            
        if hasattr(self, '_tunnel_process') and self._tunnel_process:
            try:
                logger.info("Stopping cloudflared tunnel...")
                self._tunnel_process.terminate()
                self._tunnel_process.wait(timeout=2)
            except:
                self._tunnel_process.kill()
            self._tunnel_process = None
    
    def __del__(self):
        self.close()
