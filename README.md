# Cliper

Tool lokal untuk memotong highlight otomatis dari **YouTube** atau **video lokal**. AI (DeepSeek resmi, OpenRouter sebagai cadangan) mengusulkan clip sesuai durasi yang kamu pilih. Hasil export memakai fade lembut, caption, dan hook overlay.

Pakai hanya video yang kamu punya haknya.

## Persiapan

1. Install [ffmpeg](https://ffmpeg.org/download.html) dan pastikan `ffmpeg` + `ffprobe` ada di PATH.
2. Install Python 3.11+ dan Node.js 18+.
3. Copy env:

```powershell
copy .env.example .env
```

Isi `DEEPSEEK_API_KEY` dari [platform.deepseek.com](https://platform.deepseek.com) (utama untuk highlight AI). `OPENROUTER_API_KEY` cadangan. `OPENAI_API_KEY` opsional — jika diisi, transkrip memakai Whisper API; jika kosong, memakai faster-whisper lokal.

## Jalankan

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Frontend (terminal baru):

```powershell
cd frontend
npm install
npm run dev
```

Buka http://localhost:5173

## Deploy ke Railway

Repository ini memakai satu service Railway. `Dockerfile` akan membuild frontend,
memasang FFmpeg, lalu menjalankan backend sekaligus frontend pada satu domain.

1. Push repository ke GitHub tanpa mengikutkan `env` atau `.env`.
2. Di Railway pilih **New Project > Deploy from GitHub Repo**.
3. Pilih repository ini. Railway akan membaca `railway.toml` dan memakai `Dockerfile`.
4. Tambahkan environment variables di Railway:

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=deepseek/deepseek-chat
OPENAI_API_KEY=...
WHISPER_MODEL=base
CORS_ORIGINS=https://alamat-domain-railway-kamu.up.railway.app
```

5. Setelah deploy, buka domain Railway dan cek `/api/health`.

### Catatan YouTube di Railway

YouTube kadang memblokir IP cloud Railway dengan pesan `Sign in to confirm
you're not a bot`. Untuk trial paling stabil, upload video lokal. Jika harus
mengambil YouTube, gunakan cookies dalam format Netscape melalui variable
Railway `YOUTUBE_COOKIES` atau file yang dirujuk oleh `YOUTUBE_COOKIES_FILE`.
Cookies adalah rahasia dan jangan pernah di-commit ke GitHub.

## Alur

1. Tempel URL YouTube atau upload file.
2. Pilih durasi clip (15 / 30 / 45 / 60 / custom).
3. Generate highlight.
4. Preview, edit start/end, pilih clip.
5. Export — download file mp4 yang sudah di-fade + caption.
