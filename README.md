# 🚀 TIFA Emotion AI

![TIFA Banner](https://img.shields.io/badge/TIFA%20Robot-Emotion%20AI-blueviolet.svg)

TIFA Protocol is an Emotion-aware conversational AI system designed for a Food and Beverage delivery robot. This repository acts as the main local engine that captures voice input, detects emotion natively, builds contextual LLaMA responses, synthesizes emotional text-to-speech, and transmits the resulting data real-time via WebSocket and a PostgreSQL Database.

## 🚀 Panduan Instalasi Menyeluruh: Panduan Download TIFA di Perangkat Baru

Jika Anda berada di perangkat PC/Laptop baru dan ingin merakit program TIFA Emotion AI agar berjalan tepat seperti awal pengembangkan, ikuti langkah-langkah **mulai dari nol** berikut ini.

### 📌 Tahap 1: Instalasi Software Fundamental

Anda **WAJIB** menginstall program-program dasar ini terlebih dahulu.

#### 1. Install Git
Gunanya untuk men-download file program ("clone") dari Github Anda ke perangkat lokal secara utuh.
- Download Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Install (Next > Next > Finish).

#### 2. Install Python >= 3.9
Menjalankan otak sistem AI. **Jangan sembarang install**, pastikan dicentang "Add to PATH"!
- Download dari: [https://python.org/downloads/](https://python.org/downloads/)
- **Super Penting saat menginstall**: Di layar paling awal, WAJIB centang kotak **"Add python.exe to PATH"** di pojok kiri bawah, lalu klik Install Now.

#### 3. Install Ollama (AI Offline Engine)
Bagian dari otak yang merakit kalimat respons TIFA LLaMA secara offline.
- Download dari: [https://ollama.com/download](https://ollama.com/download)
- Selesai penginstalan, buka terminal (CMD / PowerShell) dan unduh data pengetahuan *LLaMA*:
  ```bash
  ollama pull llama3.2:3b
  ```

#### 4. Install Cloudflared (Koneksi Database)
Menerobos firewall port agar database PostgreSQL TIFA dapat diakses lokal.
- Buka **PowerShell sebagai Administrator** (Klik kanan Start > Windows PowerShell (Admin))
- Copy paste skrip instalasi ini lalu tekan enter:
  ```powershell
  winget install Cloudflare.cloudflared --accept-package-agreements --accept-source-agreements
  ```
---

### 📥 Tahap 2: Download Program (Clone dari GitHub)

Langkah mengunduh repositori github ini:
1. Buka File Explorer PC baru, kreasikan folder direktori di drive bebas. Masuk ke folder tersebut.
2. Di dalam folder, klik kanan ruang kosong > Pilih **"Open in Terminal"** atau **"Git Bash Here"**.
3. *Clone* kode program ini dengan perintah (*silakan asumsikan URL berikut disesuaikan url repository github anda sendiri*):
   ```bash
   git clone https://github.com/abidrozhan/NAMA_REPOSITORI_INI.git
   ```
4. Masuk ke direktori baru bentukan git clone tersebut.
   ```bash
   cd NAMA_REPOSITORI_INI
   ```

---

### 📦 Tahap 3: Instalasi Library Module TIFA

Untuk menghindari bug dari bentrok library internal, setup *Virtual Environment* (VENV).
Pastikan terminal saat ini **sedang berada di dalam root folder projek TIFA**. Urutkan command:

```bash
python -m venv venv
venv\Scripts\activate
```
*(Saat prefix `(venv)` muncul di konsol Anda menandakan mesin virtual sudah aktif)*

Lalu pasang dependencies engine-nya:
```bash
pip install -r requirements.txt
```

---

### ⚡ Tahap 4: Setup Model Sensor Suara Lokal

Tahap final untuk men-setup sistem telinga AI `Faster-Whisper` agar model ML-nya terekstrak ke PC.
Ketik:
```bash
python setup_models.py
```
Akan ada file ukuran ringan yang otomatis di-download ke local cache `data/` dan referensi MP3 TTS akan diformat ulang. Status centang hijau semua (✅) menendakan proses sehat.

---

### ▶️ Tahap 5: PROGRAM READY!

Super!! Program selesai dirangkai!
Tatkala ingin **MENGHIDUPKAN TIFA** cukup buka terminal dalam direktori, nyalakan `venv`, dan pangil `main.py`:

```bash
venv\Scripts\activate       #(Wajib untuk masuk modul virtual jika Anda memakai Command Prompt baru)
python main.py
```

Sistem akan otomatis me-rutin *cloudflared tunnel*, membaca Database PostgreSQL, dan terhubung ke WebSocket *Robotic Remote*. Program siap di ajak berbiara lewat Microphone Anda!
