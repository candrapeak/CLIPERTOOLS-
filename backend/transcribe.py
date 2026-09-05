from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import httpx

_model = None
_model_name = ""
_model_lock = Lock()


def _whisper_model():
    global _model, _model_name
    name = os.getenv("WHISPER_MODEL", "base")
    with _model_lock:
        if _model is None or _model_name != name:
            from faster_whisper import WhisperModel

            _model = WhisperModel(name, device="cpu", compute_type="int8")
            _model_name = name
        return _model


def transcribe(audio_path: str | Path) -> list[dict]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return transcribe_openai(audio_path)
        except Exception:
            return transcribe_local(audio_path)
    return transcribe_local(audio_path)


def transcribe_local(audio_path: str | Path) -> list[dict]:
    model = _whisper_model()
    segments, _info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=3,
        word_timestamps=True,
    )
    out: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        words = []
        for word in seg.words or []:
            token = (word.word or "").strip()
            if not token:
                continue
            words.append(
                {
                    "start": float(word.start or 0),
                    "end": float(word.end or 0),
                    "word": token,
                }
            )
        out.append(
            {
                "start": float(seg.start or 0),
                "end": float(seg.end or 0),
                "text": text,
                "words": words,
            }
        )
    return out


def transcribe_openai(audio_path: str | Path) -> list[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY kosong")

    with Path(audio_path).open("rb") as handle:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
            files={"file": (Path(audio_path).name, handle, "audio/wav")},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    out: list[dict] = []
    for seg in payload.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        words = []
        for word in seg.get("words") or []:
            token = str(word.get("word") or "").strip()
            if not token:
                continue
            words.append(
                {
                    "start": float(word.get("start") or 0),
                    "end": float(word.get("end") or 0),
                    "word": token,
                }
            )
        out.append(
            {
                "start": float(seg.get("start") or 0),
                "end": float(seg.get("end") or 0),
                "text": text,
                "words": words,
            }
        )
    return out
