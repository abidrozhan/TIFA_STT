"""
NLP Intent Classifier untuk Robot TIFA
Mengklasifikasikan teks ke dalam intent: pujian, terima_kasih, tanya_nama, atau unknown
"""

# Definisi keywords untuk setiap intent
INTENT_KEYWORDS = {
    "pujian": [
        "bagus", "keren", "hebat", "mantap", "luar biasa", "top", "perfect", 
        "sempurna", "kece", "amazing", "wow", "cantik", "ganteng", "pintar",
        "cerdas", "baik", "terbaik", "jago", "sip", "oke banget", "mantul",
        "kerja bagus", "good job", "excellent", "great", "awesome", "cool",
        "luar biasa", "memukau", "menakjubkan", "fantastis"
    ],
    "terima_kasih": [
        "terima kasih", "makasih", "thanks", "thank you", "terimakasih",
        "tengkyu", "trims", "makasi", "tq", "thx", "thankyou", "thank",
        "terima kasih banyak", "makasih banyak", "thanks a lot"
    ],
    "tanya_nama": [
        "siapa nama", "namamu", "nama kamu", "siapa kamu", "kamu siapa",
        "siapa namamu", "nama mu", "siapa sih kamu", "kamu itu siapa",
        "namanya siapa", "panggil apa", "dipanggil apa", "boleh tau nama",
        "boleh tahu nama", "perkenalkan diri", "siapa dirimu"
    ]
}


def classify_intent(text):
    """
    Mengklasifikasikan teks ke dalam intent yang sesuai.
    
    Args:
        text: Teks hasil speech-to-text
        
    Returns:
        Intent string: 'pujian', 'terima_kasih', 'tanya_nama', atau 'unknown'
    """
    if not text:
        return "unknown"
    
    # Normalisasi teks (lowercase)
    text_lower = text.lower().strip()
    
    # Cek setiap intent
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            # Cek apakah keyword ada di dalam teks
            if keyword.lower() in text_lower:
                return intent
    
    # Jika tidak ada keyword yang cocok
    return "unknown"


def get_response_audio(intent):
    """
    Mendapatkan path file audio berdasarkan intent.
    
    Args:
        intent: Hasil klasifikasi intent
        
    Returns:
        Path ke file audio response
    """
    audio_mapping = {
        "pujian": "audio/response_pujian.mp3",
        "terima_kasih": "audio/response_terima_kasih.mp3",
        "tanya_nama": "audio/response_tanya_nama.mp3",
        "unknown": "audio/response_unknown.mp3"
    }
    
    return audio_mapping.get(intent, "audio/response_unknown.mp3")


# Test jika dijalankan langsung
if __name__ == "__main__":
    test_sentences = [
        "Kamu keren banget!",
        "Terima kasih ya robot",
        "Siapa nama kamu?",
        "Cuaca hari ini cerah",
        "Wah hebat sekali",
        "Makasih banyak",
        "Kamu siapa sih?",
        "Aku mau makan"
    ]
    
    print("=" * 50)
    print("TEST INTENT CLASSIFIER")
    print("=" * 50)
    
    for sentence in test_sentences:
        intent = classify_intent(sentence)
        audio = get_response_audio(intent)
        print("")
        print("Input: \"{}\"".format(sentence))
        print("  -> Intent: {}".format(intent))
        print("  -> Audio: {}".format(audio))
