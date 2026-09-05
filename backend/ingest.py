from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import UploadFile
from yt_dlp import YoutubeDL

from .ffmpeg_utils import probe_duration

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "data" / "uploads"
EXPORTS = ROOT / "data" / "exports"

ALLOWED_SUFFIX = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def ensure_dirs() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return (stem or "video")[:60]


def save_upload(job_id: str, file: UploadFile) -> tuple[Path, str, float]:
    ensure_dirs()
    raw_name = file.filename or "video.mp4"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise ValueError(f"Format tidak didukung: {suffix or 'tanpa ekstensi'}")

    dest = UPLOADS / f"{job_id}_{_safe_stem(Path(raw_name).stem)}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    duration = probe_duration(dest)
    title = Path(raw_name).stem
    return dest, title, duration


def download_youtube(job_id: str, url: str) -> tuple[Path, str, float]:
    ensure_dirs()
    if not re.search(r"youtube\.com|youtu\.be", url, re.I):
        raise ValueError("URL harus dari YouTube")

    outtmpl = str(UPLOADS / f"{job_id}_%(title).80s.%(ext)s")
    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Gagal membaca info YouTube")
        title = str(info.get("title") or "YouTube")
        prepared = ydl.prepare_filename(info)
        path = Path(prepared)
        if path.suffix.lower() != ".mp4":
            merged = path.with_suffix(".mp4")
            if merged.exists():
                path = merged

    if not path.exists():
        matches = sorted(UPLOADS.glob(f"{job_id}_*"))
        if not matches:
            raise RuntimeError("File YouTube tidak ditemukan setelah unduh")
        path = matches[0]

    duration = float(info.get("duration") or probe_duration(path))
    return path, title, duration
