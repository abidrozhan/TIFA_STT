"""
TIFA Robot - Model Setup Script
================================
Downloads and sets up all required models for the emotion-aware AI system.

Run this once before using the main application:
    python setup_models.py

This will:
1. Download wav2vec2 STT model
2. Download SpeechBrain emotion model  
3. Generate emotion reference audio samples
4. Check Ollama and LLaMA availability
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))


def check_python_version():
    """Check Python version"""
    print("=" * 60)
    print("TIFA Emotion AI - Model Setup")
    print("=" * 60)
    print()
    
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"   Current: Python {sys.version}")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")


def check_dependencies():
    """Check and install missing dependencies"""
    print("\n[Step 1] Checking dependencies...")
    
    required = [
        "torch", "transformers", "speechbrain", "librosa",
        "sounddevice", "numpy", "scipy", "edge_tts",
        "scikit-learn", "joblib", "pygame", "rich", "pydantic", "ollama"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def setup_directories():
    """Create required directories"""
    print("\n[Step 2] Setting up directories...")
    
    from tifa_emotion_ai.config import config
    config.ensure_directories()
    
    print(f"  ✓ Created {config.DATA_DIR}")
    print(f"  ✓ Created {config.MODEL_DIR}")
    print(f"  ✓ Created {config.EMOTION_SAMPLES_DIR}")
    print(f"  ✓ Created {config.TRAINING_DATA_DIR}")


def download_stt_model():
    """Download wav2vec2 STT model"""
    print("\n[Step 3] Downloading Speech-to-Text model...")
    print("  This may take a few minutes...")
    
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        from tifa_emotion_ai.config import config
        
        print(f"  Downloading: {config.STT_MODEL}")
        
        processor = Wav2Vec2Processor.from_pretrained(config.STT_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(config.STT_MODEL)
        
        print("  ✓ wav2vec2 model downloaded")
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_emotion_model():
    """Download SpeechBrain emotion model"""
    print("\n[Step 4] Downloading Emotion Recognition model...")
    print("  This may take a few minutes...")
    
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
        from tifa_emotion_ai.config import config
        
        print(f"  Downloading: {config.EMOTION_MODEL}")
        
        classifier = EncoderClassifier.from_hparams(
            source=config.EMOTION_MODEL
        )
        
        print("  ✓ Emotion model downloaded")
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Try: pip install speechbrain")
        return False


def generate_emotion_samples():
    """Generate emotion reference audio samples"""
    print("\n[Step 5] Generating emotion reference audio...")
    
    try:
        from tifa_emotion_ai.tts.emotion_voice import EmotionVoiceManager
        import asyncio
        
        manager = EmotionVoiceManager()
        
        # Run async generation
        asyncio.run(manager.generate_all_samples(overwrite=False))
        
        # Check results
        samples = manager.list_samples()
        for emotion, exists in samples.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {emotion}")
        
        return all(samples.values())
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_ollama():
    """Check Ollama installation and model"""
    print("\n[Step 6] Checking Ollama...")
    
    from tifa_emotion_ai.config import config
    
    try:
        import requests
        
        # Check if Ollama is running
        response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("  ✓ Ollama is running")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if config.OLLAMA_MODEL in model_names or \
               any(config.OLLAMA_MODEL in n for n in model_names):
                print(f"  ✓ Model '{config.OLLAMA_MODEL}' available")
                return True
            else:
                print(f"  ⚠ Model '{config.OLLAMA_MODEL}' not found")
                print(f"    Run: ollama pull {config.OLLAMA_MODEL}")
                return False
        else:
            print("  ✗ Ollama not responding")
            return False
    
    except requests.ConnectionError:
        print("  ✗ Ollama not running")
        print("    1. Install Ollama from https://ollama.ai")
        print("    2. Run: ollama serve")
        print(f"    3. Run: ollama pull {config.OLLAMA_MODEL}")
        return False
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_tts():
    """Check TTS availability"""
    print("\n[Step 7] Checking TTS engines...")
    
    # Check Edge TTS
    try:
        import edge_tts
        print("  ✓ Edge TTS available")
    except ImportError:
        print("  ✗ Edge TTS not installed")
        print("    Run: pip install edge-tts")
    
    # Check Coqui TTS (optional)
    try:
        from TTS.api import TTS
        print("  ✓ Coqui TTS available (optional)")
    except ImportError:
        print("  ⚠ Coqui TTS not installed (optional, for better quality)")
        print("    Run: pip install TTS")


def main():
    """Main setup function"""
    check_python_version()
    
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first.")
        sys.exit(1)
    
    setup_directories()
    
    stt_ok = download_stt_model()
    emotion_ok = download_emotion_model()
    samples_ok = generate_emotion_samples()
    ollama_ok = check_ollama()
    check_tts()
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    results = [
        ("Speech-to-Text", stt_ok),
        ("Emotion Recognition", emotion_ok),
        ("Reference Audio", samples_ok),
        ("Ollama/LLaMA", ollama_ok),
    ]
    
    all_ok = True
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("✅ All components ready!")
        print("\nRun the application with:")
        print("   python main.py")
    else:
        print("⚠️  Some components need attention.")
        print("   Review the errors above and resolve them.")
    
    print()


if __name__ == "__main__":
    main()
