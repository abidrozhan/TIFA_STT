"""
==================================================================================
TIFA Robot - Emotion-Aware Conversational AI (Robot Pengantar Makanan & Minuman)
==================================================================================

File ini adalah ENTRY POINT utama untuk menjalankan seluruh sistem TIFA.

CARA MENJALANKAN:
   python main.py              # Jalankan percakapan interaktif
   python main.py --setup      # Setup dan download model (jalankan pertama kali)
   python main.py --light      # Mode ringan untuk device lambat

ALUR KERJA PIPELINE:
   1. Rekam suara pengguna dari mikrofon
   2. Konversi suara ke teks (wav2vec2 - offline, tidak perlu internet)
   3. Deteksi emosi dari audio (SpeechBrain)
   4. Klasifikasi emosi final (trainable classifier - belajar terus-menerus)
   5. Generate respons dengan LLaMA (via Ollama - lokal di PC)
   6. Sintesis suara dengan emosi yang sesuai (Coqui/Edge TTS)
   7. Putar audio respons ke speaker

PERSYARATAN:
   - Python 3.9+
   - Ollama terinstall dan berjalan (ollama serve)
   - Model LLaMA sudah di-pull (ollama pull llama3.2:3b)
   - Mikrofon dan speaker

Author: TIFA Team
Version: 2.0.0
"""

import sys
import io

# Force UTF-8 encoding for Windows console (to support emojis)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import time
import pygame
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

import os
from tifa_emotion_ai.config import config
from tifa_emotion_ai.utils import (
    setup_logging, print_header, print_status, 
    print_emotion, console, format_duration
)
from tifa_emotion_ai.stt import Wav2VecSTT, AudioProcessor
from tifa_emotion_ai.emotion import SpeechEmotionRecognizer, EmotionClassifier, EmotionDataset
from tifa_emotion_ai.llm import OllamaClient
from tifa_emotion_ai.tts import EmotionTTS
from tifa_emotion_ai.ws_client import TIFAWebSocketClient


class TIFAEmotionAI:
    """
    ============================================================================
    KELAS UTAMA SISTEM TIFA - Emotion-Aware AI
    ============================================================================
    
    Kelas ini mengintegrasikan semua komponen AI menjadi satu sistem:
    
    KOMPONEN YANG DIGUNAKAN:
    - STT (Speech-to-Text)     : wav2vec2 - konversi suara ke teks (offline)
    - Emotion Recognition      : SpeechBrain - deteksi emosi dari audio
    - Emotion Classification   : Classifier trainable - belajar dari interaksi
    - Response Generation      : LLaMA via Ollama - generate respons cerdas
    - TTS (Text-to-Speech)     : Coqui/Edge TTS - sintesis suara dengan emosi
    
    CARA PAKAI:
        tifa = TIFAEmotionAI()           # Inisialisasi sistem
        tifa.run_conversation_loop()     # Mulai percakapan loop
    """
    
    def __init__(self, use_light_mode: bool = False):
        """
        Initialize TIFA system.
        
        Args:
            use_light_mode: Use lighter models for slower devices
        """
        self.logger = setup_logging()
        self.use_light_mode = use_light_mode
        
        # Initialize pygame for audio playback (fallback)
        pygame.mixer.init()
        
        # Ensure directories exist
        config.ensure_directories()
        
        # Initialize WebSocket client for remote UI
        self.ws_client = TIFAWebSocketClient()
        self._ws_connected = False
        
        # Initialize components (lazy loading)
        self._stt = None
        self._audio_processor = None
        self._emotion_recognizer = None
        self._classifier = None
        self._dataset = None
        self._llm = None
        self._tts = None
        
        # Session stats
        self.interaction_count = 0
        self.session_start = time.time()
    
    @property
    def stt(self) -> Wav2VecSTT:
        if self._stt is None:
            console.print("[dim]Loading Speech-to-Text model...[/]")
            self._stt = Wav2VecSTT()
        return self._stt
    
    @property
    def audio_processor(self) -> AudioProcessor:
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor()
        return self._audio_processor
    
    @property
    def emotion_recognizer(self) -> SpeechEmotionRecognizer:
        if self._emotion_recognizer is None:
            console.print("[dim]Loading Emotion Recognition model...[/]")
            self._emotion_recognizer = SpeechEmotionRecognizer()
        return self._emotion_recognizer
    
    @property
    def classifier(self) -> EmotionClassifier:
        if self._classifier is None:
            self._classifier = EmotionClassifier()
        return self._classifier
    
    @property
    def dataset(self) -> EmotionDataset:
        if self._dataset is None:
            self._dataset = EmotionDataset()
        return self._dataset
    
    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = OllamaClient()
        return self._llm
    
    @property
    def tts(self) -> EmotionTTS:
        if self._tts is None:
            console.print("[dim]Loading TTS engine...[/]")
            self._tts = EmotionTTS(use_coqui=not self.use_light_mode)
        return self._tts
    
    def process_interaction(self) -> bool:
        """
        ====================================================================
        PROSES SATU SIKLUS INTERAKSI LENGKAP
        ====================================================================
        
        Fungsi ini menjalankan satu siklus penuh: 
        rekam → transcribe → deteksi emosi → generate respons → putar audio
        
        Returns:
            True jika berhasil dan lanjut, False jika harus berhenti
        """
        try:
            # Step 1: Record audio
            console.print("\n[bold cyan]🎤 Listening...[/]")
            audio = self.audio_processor.record_from_mic()
            
            if audio is None or len(audio) < 1000:  # Too short
                # console.print("[yellow]Tidak terdengar, coba lagi...[/]")
                return True
            
            # Step 2: Transcribe
            console.print("[dim]Transcribing...[/]")
            text = self.stt.transcribe(audio)
            
            if not text.strip():
                console.print("[yellow]Tidak dapat mengenali ucapan (hening/noise)[/]")
                return True
            
            console.print(f"[bold white]📝 \"{text}\"[/]")
            
            # Step 3: Detect emotion from audio AND text
            audio_emotion, emotion_confidence = self.emotion_recognizer.predict(audio, text=text)
            
            # Step 4: Classify final emotion
            if self.classifier.is_trained:
                final_emotion, class_confidence = self.classifier.predict_with_confidence(
                    text, audio_emotion
                )
                confidence = (emotion_confidence + class_confidence) / 2
            else:
                final_emotion = audio_emotion
                confidence = emotion_confidence
            
            print_emotion(final_emotion, confidence)
            
            # Step 5: Generate response with LLaMA
            console.print("[dim]Generating response...[/]")
            start_gen = time.time()
            
            if self.llm._check_connection():
                response = self.llm.generate_response(text, final_emotion, confidence)
            else:
                console.print("[red]Ollama tidak terhubung![/]")
                response = "Maaf, sistem otak saya sedang tidak terhubung. Mohon cek koneksi Ollama."
            
            gen_time = time.time() - start_gen
            
            console.print(f"[bold green]🤖 {response}[/]")
            console.print(f"[dim](Generated in {format_duration(gen_time)})[/]")
            
            # Step 6: Synthesize and send/play response
            console.print("[dim]Synthesizing speech...[/]")
            
            # Use unique filename - use .wav for WebSocket compatibility
            timestamp = int(time.time() * 1000)
            output_file = config.DATA_DIR / f"response_{timestamp}.wav"
            
            audio_path = self.tts.synthesize(response, final_emotion, 
                                             output_path=str(output_file))
            
            if audio_path:
                # Try to send via WebSocket to remote UI
                ws_sent = self._send_audio_ws(audio_path, final_emotion)
                
                if not ws_sent:
                    # Fallback: play locally
                    self._play_audio(audio_path)
            
            # Step 7: Update training data (continuous learning)
            self.dataset.add_sample(text, audio_emotion, final_emotion)
            self.classifier.partial_train(text, audio_emotion, final_emotion)
            
            self.interaction_count += 1
            
            return True
        
        except KeyboardInterrupt:
            return False
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            self.logger.exception("Interaction error")
            return True
    
    def _play_audio(self, audio_path: str):
        """Play audio file locally and clean up (fallback)"""
        try:
            if not os.path.exists(audio_path):
                return

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Unload to release file lock
            pygame.mixer.music.unload()
            
            # Delete temp file
            try:
                os.remove(audio_path)
            except Exception as e:
                self.logger.warning(f"Failed to delete temp file: {e}")
                
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
    
    def _send_audio_ws(self, audio_path: str, emotion: str) -> bool:
        """
        Send audio via WebSocket to remote UI.
        
        Args:
            audio_path: Path to WAV audio file
            emotion: Detected emotion for expression
            
        Returns:
            True if sent successfully
        """
        try:
            if not self._ws_connected:
                return False
            
            # First, send expression
            self.ws_client.send_expression(emotion)
            
            # Then send audio with expression
            success = self.ws_client.send_audio_with_expression(audio_path, emotion)
            
            if success:
                console.print(f"[bold cyan]📡 Audio sent to UI via WebSocket[/]")
                # Clean up temp file after successful send
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            return success
            
        except Exception as e:
            self.logger.error(f"WebSocket send error: {e}")
            return False
    
    def run_conversation_loop(self):
        """
        ====================================================================
        LOOP PERCAKAPAN UTAMA
        ====================================================================
        
        Fungsi ini menjalankan loop percakapan interaktif yang terus-menerus
        sampai pengguna menekan Ctrl+C untuk berhenti.
        """
        # Tampilkan header dan info sistem
        print_header(
            "TIFA Robot - Emotion-Aware AI v2.0",
            "Powered by wav2vec2, SpeechBrain, LLaMA, Coqui TTS"
        )
        
        # Connect WebSocket
        console.print("[dim]Connecting to WebSocket UI...[/]")
        self._ws_connected = self.ws_client.connect()
        if self._ws_connected:
            console.print("[bold green]📡 WebSocket connected! Audio will be sent to remote UI.[/]")
        else:
            console.print("[yellow]⚠ WebSocket not available. Audio will play locally.[/]")
        
        # Tampilkan komponen yang digunakan
        console.print("[bold]Komponen:[/]")
        console.print(f"  • STT: wav2vec2")
        console.print(f"  • Emotion: SpeechBrain + Trainable Classifier")
        console.print(f"  • LLM: {config.OLLAMA_MODEL}")
        console.print(f"  • TTS: {self.tts.engine_name}")
        ws_status = "WebSocket (Remote UI)" if self._ws_connected else "Local Speaker"
        console.print(f"  • Output: {ws_status}")
        console.print()
        console.print("[yellow]Tekan Ctrl+C untuk berhenti[/]")
        console.print("-" * 60)
        
        try:
            while True:
                try:
                    if not self.process_interaction():
                        break
                except Exception as e:
                    console.print(f"[red]Error in loop: {e}. Retrying...[/]")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup on exit"""
        console.print("\n")
        console.print("[yellow]Menyimpan data...[/]")
        
        # Close WebSocket
        if self._ws_connected:
            self.ws_client.close()
            console.print("[dim]WebSocket disconnected[/]")
        
        # Save training data
        self.dataset.save()
        
        # Save classifier
        self.classifier.save()
        
        # Save conversation
        self.llm.context.save_session()
        
        # Show stats
        duration = time.time() - self.session_start
        console.print()
        console.print("[bold]📊 Session Stats:[/]")
        console.print(f"  • Interactions: {self.interaction_count}")
        console.print(f"  • Duration: {format_duration(duration)}")
        console.print(f"  • Training samples: {len(self.dataset)}")
        console.print()
        console.print("[bold green]Sampai jumpa! 👋[/]")
        
        pygame.mixer.quit()
    
    def correct_emotion(self, text: str, audio_emotion: str, correct_emotion: str):
        """
        Allow user to correct emotion classification.
        This helps the classifier learn better.
        
        Args:
            text: Original transcribed text
            audio_emotion: Originally detected audio emotion
            correct_emotion: User's corrected emotion label
        """
        self.dataset.add_correction(text, audio_emotion, correct_emotion)
        self.classifier.partial_train(text, audio_emotion, correct_emotion)
        console.print(f"[green]✓ Learned: '{text[:30]}...' -> {correct_emotion}[/]")


def setup_system():
    """
    ====================================================================
    SETUP PERTAMA KALI
    ====================================================================
    
    Jalankan fungsi ini sekali sebelum menggunakan sistem untuk:
    - Mengecek apakah Ollama berjalan
    - Download model wav2vec2 dan SpeechBrain
    - Generate audio referensi untuk TTS
    
    Cara menjalankan: python main.py --setup
    """
    print_header("TIFA Setup", "Downloading models and generating samples")
    
    config.ensure_directories()
    
    console.print("[bold]Step 1: Checking Ollama...[/]")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            console.print("[green]  ✓ Ollama is running[/]")
        else:
            console.print("[red]  ✗ Ollama not responding[/]")
            console.print("    Run: ollama serve")
    except:
        console.print("[red]  ✗ Ollama not running[/]")
        console.print("    Install from: https://ollama.ai")
        console.print(f"    Then run: ollama pull {config.OLLAMA_MODEL}")
    
    console.print("\n[bold]Step 2: Pre-loading models...[/]")
    console.print("  This may take a few minutes on first run...")
    
    # Pre-download models by importing
    try:
        console.print("  Loading wav2vec2...")
        from tifa_emotion_ai.stt import Wav2VecSTT
        stt = Wav2VecSTT()
        console.print("[green]  ✓ wav2vec2 ready[/]")
    except Exception as e:
        console.print(f"[red]  ✗ wav2vec2 error: {e}[/]")
    
    try:
        console.print("  Loading SpeechBrain emotion model...")
        from tifa_emotion_ai.emotion import SpeechEmotionRecognizer
        emo = SpeechEmotionRecognizer()
        console.print("[green]  ✓ Emotion model ready[/]")
    except Exception as e:
        console.print(f"[red]  ✗ Emotion model error: {e}[/]")
    
    console.print("\n[bold]Step 3: Generating emotion reference audio...[/]")
    try:
        from tifa_emotion_ai.tts import EmotionTTS
        tts = EmotionTTS(use_coqui=False)  # Use Edge TTS for reference
        tts.setup_references()
        console.print("[green]  ✓ Reference audio generated[/]")
    except Exception as e:
        console.print(f"[red]  ✗ Reference audio error: {e}[/]")
    
    console.print("\n[bold green]Setup complete![/]")
    console.print("\nRun with: [cyan]python main.py[/]")


def main():
    """
    ====================================================================
    FUNGSI MAIN - ENTRY POINT APLIKASI
    ====================================================================
    
    Fungsi ini akan dipanggil ketika menjalankan: python main.py
    
    Pilihan argumen:
        --setup : Jalankan setup pertama kali (download model, dll)
        --light : Gunakan mode ringan untuk device yang lambat
    """
    parser = argparse.ArgumentParser(
        description="TIFA Robot - Emotion-Aware Conversational AI"
    )
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Run first-time setup"
    )
    parser.add_argument(
        "--light",
        action="store_true", 
        help="Use lighter models (slower devices)"
    )
    
    args = parser.parse_args()
    
    if args.setup:
        setup_system()
    else:
        tifa = TIFAEmotionAI(use_light_mode=args.light)
        tifa.run_conversation_loop()


if __name__ == "__main__":
    main()
