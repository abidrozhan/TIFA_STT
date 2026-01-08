"""
Script untuk generate audio responses menggunakan Edge TTS
Suara ceria untuk robot pelayanan TIFA
Run script ini sekali untuk membuat file audio
"""

import edge_tts
import asyncio
import os

# Pilihan suara Indonesia dari Edge TTS
# - id-ID-ArdiNeural (pria) - lebih serius/tegas
# - id-ID-GadisNeural (wanita) - lebih ramah/ceria
VOICE = "id-ID-GadisNeural"  # Suara wanita lebih cocok untuk pelayanan

# ============================================================
# PENGATURAN GAYA SUARA CERIA (untuk robot pelayanan)
# ============================================================
# Rate: kecepatan bicara
#   - "+5%" = sedikit lebih cepat → terdengar antusias
RATE = "+10%"

# Pitch: tinggi rendah nada  
#   - "+5Hz" = sedikit lebih tinggi → terdengar ceria & ramah
PITCH = "+7Hz"

# Volume: keras lemah suara
VOLUME = "+0%"
# ============================================================

# Buat folder audio jika belum ada
audio_folder = "audio"
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)
    print("[OK] Folder '{}' berhasil dibuat".format(audio_folder))

# Daftar response untuk setiap intent (dengan kata-kata lebih ramah)
responses = {
    "response_pujian": "Wah, terima kasih banyak atas pujiannya yaa!",
    "response_terima_kasih": "Sama sama! Semoga harimu menyenangkan yaa!",
    "response_tanya_nama": "Halo! Nama aku TIFA, robot pintar pengantar makanan!",
    "response_unknown": "Maaf ya, aku belum mengerti maksudnya. Bisakah kamu ulangi?"
}

async def generate_audio(text, filepath):
    """Generate audio menggunakan Edge TTS dengan pengaturan ceria"""
    communicate = edge_tts.Communicate(
        text, 
        VOICE,
        rate=RATE,
        pitch=PITCH,
        volume=VOLUME
    )
    await communicate.save(filepath)

async def main():
    print("Generating audio files with Edge TTS...")
    print("Voice: {} (Suara Ceria)".format(VOICE))
    print("Rate: {} | Pitch: {} | Volume: {}".format(RATE, PITCH, VOLUME))
    print("")
    
    for filename, text in responses.items():
        filepath = os.path.join(audio_folder, "{}.mp3".format(filename))
        
        # Generate audio dengan Edge TTS
        await generate_audio(text, filepath)
        
        print("[OK] Generated: {}".format(filepath))
    
    print("")
    print("[DONE] Semua audio berhasil di-generate!")
    print("Lokasi: {}".format(os.path.abspath(audio_folder)))

if __name__ == "__main__":
    asyncio.run(main())
