"""
Speech-to-Text dengan NLP Intent Classification untuk Robot TIFA
Mengklasifikasikan ucapan dan merespon dengan audio yang sesuai
"""

import speech_recognition as sr
import pygame
import os
import time

from intent_classifier import classify_intent, get_response_audio

# Inisialisasi
r = sr.Recognizer()
pygame.mixer.init()

def play_audio(audio_path):
    """Memainkan file audio menggunakan pygame"""
    if not os.path.exists(audio_path):
        print("[WARNING] Audio file tidak ditemukan: {}".format(audio_path))
        print("          Jalankan 'python generate_audio.py' terlebih dahulu")
        return
    
    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        
        # Tunggu sampai audio selesai
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print("[ERROR] Error playing audio: {}".format(e))

def record_text():
    """Merekam dan mengkonversi speech ke text"""
    while True:
        try:
            with sr.Microphone() as source2:
                print("")
                print("[MIC] Listening...")
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2, language="id-ID")
                return MyText

        except sr.RequestError as e:
            print("[ERROR] Cloud not request result: {}".format(e))

        except sr.UnknownValueError:
            print("[INFO] Tidak terdengar, coba lagi...")

def output_text(text):
    """Menyimpan teks ke file output"""
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

def process_speech(text):
    """Memproses teks: klasifikasi intent dan putar audio response"""
    # Klasifikasi intent menggunakan NLP
    intent = classify_intent(text)
    
    # Dapatkan path audio untuk intent tersebut
    audio_path = get_response_audio(intent)
    
    print("[TEXT] \"{}\"".format(text))
    print("[INTENT] {}".format(intent))
    print("[AUDIO] Playing: {}".format(audio_path))
    
    # Putar audio response
    play_audio(audio_path)
    
    return intent

# Main program
if __name__ == "__main__":
    print("=" * 60)
    print("ROBOT TIFA - Speech Recognition + NLP Intent Classifier")
    print("=" * 60)
    print("")
    print("Intent yang dikenali:")
    print("  * PUJIAN       -> 'keren', 'bagus', 'hebat', dll")
    print("  * TERIMA_KASIH -> 'terima kasih', 'makasih', dll")
    print("  * TANYA_NAMA   -> 'siapa nama kamu', 'namamu siapa', dll")
    print("")
    print("Tekan Ctrl+C untuk berhenti")
    print("-" * 60)
    
    # Cek apakah folder audio sudah ada
    if not os.path.exists("audio"):
        print("")
        print("[WARNING] Folder 'audio' tidak ditemukan!")
        print("          Jalankan 'python generate_audio.py' terlebih dahulu")
        print("-" * 60)
    
    try:
        while True:
            # Rekam suara dan konversi ke teks
            text = record_text()
            
            # Simpan ke file
            output_text(text)
            
            # Proses: klasifikasi + response audio
            process_speech(text)
            
    except KeyboardInterrupt:
        print("")
        print("")
        print("Program dihentikan. Sampai jumpa!")
        pygame.mixer.quit()
