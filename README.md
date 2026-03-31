# TIFA Robot - Emotion-Aware Conversational AI

<div align="center">

**Sistem AI percakapan dengan kesadaran emosi untuk Robot TIFA**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 Fitur Utama

- **🎤 Speech-to-Text** - wav2vec2 tanpa dependency Google (offline)
- **😊 Emotion Recognition** - Deteksi emosi dari suara (SpeechBrain)
- **🧠 Continuous Learning** - Classifier yang belajar terus-menerus
- **🦙 LLaMA Integration** - Response generation via Ollama (lokal)
- **🔊 Emotion TTS** - Suara yang sesuai dengan emosi (Coqui/Edge)

## 📋 Requirements

- Python 3.9+
- [Ollama](https://ollama.ai) (untuk LLaMA)
- Mikrofon
- GPU opsional (lebih cepat dengan GPU)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Install Ollama dari https://ollama.ai
# Kemudian run:
ollama serve
ollama pull llama3.2:3b
```

### 3. Setup Models

```bash
python setup_models.py
```

### 4. Run Application

```bash
python main.py
```

## 📁 Project Structure

```
d:/Speech TT V2/
├── main.py                    # Main application
├── setup_models.py            # Model download script
├── requirements.txt           # Dependencies
│
├── tifa_emotion_ai/           # Main package
│   ├── config.py              # Configuration
│   ├── utils.py               # Utilities
│   │
│   ├── stt/                   # Speech-to-Text
│   │   ├── wav2vec_stt.py     # wav2vec2 implementation
│   │   └── audio_processor.py # Audio preprocessing
│   │
│   ├── emotion/               # Emotion module
│   │   ├── speech_emotion.py  # SpeechBrain recognizer
│   │   ├── classifier.py      # Trainable classifier
│   │   └── dataset.py         # Training data manager
│   │
│   ├── llm/                   # LLaMA integration
│   │   ├── ollama_client.py   # Ollama API wrapper
│   │   ├── prompts.py         # Prompt templates
│   │   └── context.py         # Conversation context
│   │
│   └── tts/                   # Text-to-Speech
│       ├── coqui_tts.py       # Coqui/Edge TTS
│       └── emotion_voice.py   # Emotion voice manager
│
└── data/                      # Data storage
    ├── emotion_samples/       # Reference audio
    ├── training_data/         # ML training data
    └── models/                # Saved classifiers
```

## 🔄 Pipeline Flow

```
┌─────────────────┐
│   🎤 Microphone │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  wav2vec2 STT   │────▶│  SpeechBrain    │
│   (text)        │     │  (audio emotion)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│     Trainable Emotion Classifier        │
│        (combines text + audio)          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         LLaMA via Ollama                │
│  (generates emotion-aware response)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    Coqui/Edge TTS with Emotion          │
│   (synthesizes with matching voice)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
            🔊 Speaker
```

## 😊 Supported Emotions

| Emotion | ID | Description |
|---------|----|----|
| 😐 Neutral | netral | Ekspresi normal |
| 😊 Happy | senang | Gembira, antusias |
| 😢 Sad | sedih | Kecewa, murung |
| 😠 Angry | marah | Frustrasi, kesal |
| 😨 Fear | takut | Khawatir, cemas |
| 😲 Surprise | terkejut | Kaget, amazed |
| 🤢 Disgust | jijik | Tidak suka |

## 🧠 Continuous Learning

Sistem ini menggunakan **incremental learning** dimana classifier terus belajar dari interaksi:

```python
# Koreksi emosi manual
tifa = TIFAEmotionAI()
tifa.correct_emotion(
    text="Aku sangat lelah hari ini",
    audio_emotion="neutral",
    correct_emotion="sad"
)
# Classifier akan belajar dari koreksi ini
```

## ⚙️ Configuration

Edit `tifa_emotion_ai/config.py` untuk mengubah:

- Model paths
- Ollama settings
- TTS voice
- Audio parameters

## 📝 Usage Examples

### Basic Conversation

```python
from main import TIFAEmotionAI

tifa = TIFAEmotionAI()
tifa.run_conversation_loop()
```

### Single Interaction

```python
# Record and process single interaction
audio = tifa.audio_processor.record_from_mic()
text = tifa.stt.transcribe(audio)
emotion, confidence = tifa.emotion_recognizer.predict(audio)
response = tifa.llm.generate_response(text, emotion)
tifa.tts.synthesize(response, emotion, "output.mp3")
```

### Light Mode (Slower Devices)

```bash
python main.py --light
```

## 🐛 Troubleshooting

### Ollama tidak berjalan
```bash
ollama serve
```

### Model tidak ditemukan
```bash
ollama pull llama3.2:3b
```

### Audio tidak terdengar
- Pastikan mikrofon connected
- Check volume settings

### Error CUDA/GPU
- System akan fallback ke CPU otomatis
- Jalankan dengan `--light` untuk mode ringan

## 📄 License

MIT License - Bebas digunakan dan dimodifikasi.

---

<div align="center">
Made with ❤️ for Robot TIFA
</div>
