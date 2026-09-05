from __future__ import annotations

import os
import re
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field

from .ffmpeg_utils import export_soft_clip, extract_audio
from .highlights import suggest_clips
from .ingest import EXPORTS, download_youtube, save_upload
from .jobs import store
from .style import TEMPLATES, ClipStyle
from .transcribe import transcribe

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "env")

app = FastAPI(title="Cliper")
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YoutubeIn(BaseModel):
    url: str


class AnalyzeIn(BaseModel):
    job_id: str
    target_duration: int = Field(ge=8, le=180)
    clip_count: int = Field(default=3, ge=1, le=8)


class ClipIn(BaseModel):
    id: str
    title: str = ""
    hook: str = ""
    start: float
    end: float
    selected: bool = True


class StyleIn(BaseModel):
    template: str = "viral"
    font: str = "Arial Black"
    position: str = "center"
    active_color: str = "#FFFF00"
    idle_color: str = "#F0F0F0"
    caption_size: str = "large"
    crop: str = "tight"
    quality: str = "fast"
    watermark: str = ""
    zoom_punch: bool = True
    hook: bool = False
    fade: bool = True
    safe_margin: int = 120
    blur_amount: int = Field(default=55, ge=0, le=100)


class FrameIn(BaseModel):
    job_id: str
    start: float
    end: float


class ExportIn(BaseModel):
    job_id: str
    clips: list[ClipIn]
    fade: bool = True
    caption: bool = True
    hook: bool = False
    orientation: Literal["portrait", "landscape"] = "landscape"
    focus_speaker: bool = True
    portrait_mode: Literal["full", "letterbox", "blur"] = "full"
    style: StyleIn | None = None


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")[:40] or "clip"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "cliper"}


@app.get("/api/styles")
def list_styles() -> dict:
    return {"templates": TEMPLATES}


@app.post("/api/frame")
def frame_clip(body: FrameIn) -> dict:
    try:
        job = store.require(body.job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job tidak ditemukan") from exc
    if not job.video_path:
        raise HTTPException(400, "Video belum siap")
    from .tracker import track_speaker

    start = max(0.0, float(body.start))
    end = max(start + 0.6, float(body.end))
    track = track_speaker(
        job.video_path,
        start,
        min(end, start + 14.0),
        "portrait",
        audio_path=job.audio_path or None,
        crop_factor=2.2,
        sample_step=0.12,
    )
    if not track:
        return {"x": 0.5, "locked": False, "points": []}
    points = [{"t": round(t, 3), "x": round(x, 3)} for t, x in track.focus_points()]
    return {"x": points[0]["x"] if points else 0.5, "locked": True, "points": points}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return store.require(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, "Job tidak ditemukan") from exc


@app.post("/api/ingest/youtube")
def ingest_youtube(body: YoutubeIn) -> dict:
    job = store.create(status="downloading", step="Mengunduh YouTube", title="YouTube")
    try:
        path, title, duration = download_youtube(job.id, body.url.strip())
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        raise HTTPException(400, str(exc)) from exc
    job.video_path = str(path)
    job.title = title
    job.duration = duration
    job.status = "idle"
    job.step = "Video siap"
    return job.public()


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict:
    job = store.create(status="uploading", step="Menyimpan file", title=file.filename or "Video")
    try:
        path, title, duration = save_upload(job.id, file)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        raise HTTPException(400, str(exc)) from exc
    job.video_path = str(path)
    job.title = title
    job.duration = duration
    job.status = "idle"
    job.step = "Video siap"
    return job.public()


def _analyze_job(job_id: str, target_duration: int, clip_count: int) -> None:
    job = store.require(job_id)
    try:
        job.status = "working"
        job.error = None
        audio = Path(job.video_path).with_suffix(".wav")
        have_transcript = bool(job.segments) and bool(job.audio_path) and Path(job.audio_path).exists()
        if have_transcript:
            job.step = "Pakai transkrip tersimpan"
        else:
            job.step = "Mengekstrak audio"
            extract_audio(job.video_path, audio)
            job.audio_path = str(audio)
            job.step = "Mentranskrip audio"
            job.segments = transcribe(audio)

        same_params = (
            job.last_target_duration == target_duration
            and job.last_clip_count == clip_count
            and bool(job.clips)
        )
        if same_params:
            job.status = "ready"
            job.step = "Clip sudah siap (cache)"
            return

        job.step = "AI memilih highlight"
        job.clips = suggest_clips(
            job.segments,
            job.duration,
            target_duration,
            job.audio_path,
            clip_count,
        )
        job.last_target_duration = target_duration
        job.last_clip_count = clip_count
        job.status = "ready"
        job.step = "AI sudah pilih clip terbaik"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.step = "Gagal"


@app.post("/api/analyze")
def analyze(body: AnalyzeIn) -> dict:
    try:
        job = store.require(body.job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job tidak ditemukan") from exc
    if not job.video_path:
        raise HTTPException(400, "Video belum siap")
    if job.status == "working":
        return job.public()

    thread = Thread(
        target=_analyze_job,
        args=(job.id, body.target_duration, body.clip_count),
        daemon=True,
    )
    thread.start()
    job.status = "working"
    job.step = "Memulai analisis"
    return job.public()


def _export_job(
    job_id: str,
    clips: list[ClipIn],
    fade: bool,
    caption: bool,
    hook: bool,
    orientation: str,
    focus_speaker: bool,
    style: ClipStyle,
    portrait_mode: str,
) -> None:
    job = store.require(job_id)
    try:
        job.status = "exporting"
        job.exports = []
        selected = [c for c in clips if c.selected and c.end > c.start]
        if not selected:
            raise RuntimeError("Tidak ada clip yang dipilih")

        out_dir = EXPORTS / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        exports: list[dict[str, str]] = []
        for index, clip in enumerate(selected, start=1):
            job.step = f"Lock speaker + export {index}/{len(selected)}"
            filename = f"{index:02d}_{_safe_name(clip.title or clip.id)}.mp4"
            dest = out_dir / filename
            export_soft_clip(
                job.video_path,
                dest,
                clip.start,
                clip.end,
                hook=clip.hook or clip.title,
                segments=job.segments,
                fade=fade,
                caption=caption,
                hook_overlay=hook,
                orientation=orientation,
                focus_speaker=focus_speaker,
                style=style,
                audio_path=job.audio_path or None,
                portrait_mode=portrait_mode,
            )
            exports.append(
                {
                    "name": filename,
                    "url": f"/api/exports/{job.id}/{filename}",
                }
            )
        job.exports = exports
        job.status = "ready"
        job.step = "Export selesai"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.step = "Export gagal"


@app.post("/api/export")
def export_clips(body: ExportIn) -> dict:
    try:
        job = store.require(body.job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job tidak ditemukan") from exc
    if job.status == "exporting":
        return job.public()

    style = ClipStyle.from_payload(body.style.model_dump() if body.style else None)
    thread = Thread(
        target=_export_job,
        args=(
            job.id,
            body.clips,
            body.fade,
            body.caption,
            body.hook,
            body.orientation,
            body.focus_speaker,
            style,
            body.portrait_mode,
        ),
        daemon=True,
    )
    thread.start()
    job.status = "exporting"
    job.step = "Menyiapkan export"
    return job.public()


@app.get("/api/media/{job_id}")
def media(job_id: str) -> FileResponse:
    try:
        job = store.require(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job tidak ditemukan") from exc
    path = Path(job.video_path)
    if not path.exists():
        raise HTTPException(404, "File video tidak ada")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/exports/{job_id}/{filename}")
def download_export(job_id: str, filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(400, "Nama file tidak valid")
    path = EXPORTS / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File export tidak ada")
    return FileResponse(path, media_type="video/mp4", filename=filename)


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
