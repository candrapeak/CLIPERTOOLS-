from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


def compact_transcript(segments: list[dict], max_chars: int = 12000) -> str:
    lines: list[str] = []
    used = 0
    for seg in segments:
        line = f"[{_fmt(seg['start'])}-{_fmt(seg['end'])}] {seg['text']}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def snap_bounds(
    start: float,
    end: float,
    segments: list[dict],
    video_duration: float,
    target_duration: float,
) -> tuple[float, float]:
    start = max(0.0, float(start))
    end = min(video_duration, float(end))
    if end <= start:
        end = min(video_duration, start + target_duration)

    if segments:
        snapped_start = min(segments, key=lambda s: abs(float(s["start"]) - start))
        snapped_end = min(segments, key=lambda s: abs(float(s["end"]) - end))
        start = max(0.0, float(snapped_start["start"]))
        end = min(video_duration, float(snapped_end["end"]))
        if end <= start:
            end = min(video_duration, start + target_duration)

    # Keep close to requested duration without cutting mid-word more than needed
    if end - start > target_duration + 4:
        end = min(video_duration, start + target_duration)
    if end - start < max(6.0, target_duration * 0.55):
        end = min(video_duration, start + target_duration)
    return round(start, 3), round(end, 3)


def fallback_clips(
    video_duration: float,
    target_duration: float,
    audio_path: str | None = None,
    clip_count: int = 3,
) -> list[dict[str, Any]]:
    energy = (
        energy_clips(audio_path, video_duration, target_duration, clip_count)
        if audio_path
        else []
    )
    if energy:
        return _take_best(energy, clip_count)
    end = min(video_duration, max(1.0, target_duration))
    return _take_best(
        [
            {
                "id": "clip-1",
                "title": "Momen terbaik",
                "reason": "Dipilih otomatis dari awal video",
                "hook": "",
                "start": 0.0,
                "end": round(end, 3),
                "score": 0.4,
                "selected": True,
            }
        ],
        clip_count,
    )


def energy_clips(
    audio_path: str | None,
    video_duration: float,
    target_duration: float,
    max_clips: int = 3,
) -> list[dict[str, Any]]:
    if not audio_path:
        return []
    try:
        import wave

        import numpy as np

        with wave.open(audio_path, "rb") as wav:
            rate = wav.getframerate()
            frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if frames.size == 0:
            return []
        hop = max(1, rate // 2)
        rms = [
            float(np.sqrt(np.mean(frames[i : i + hop].astype(np.float32) ** 2)))
            for i in range(0, max(hop, frames.size - hop), hop)
        ]
        bins = max(1, int(round(target_duration / 0.5)))
        ranked: list[tuple[float, float]] = []
        for index in range(0, max(1, len(rms) - bins + 1)):
            ranked.append((sum(rms[index : index + bins]), index * 0.5))
        ranked.sort(reverse=True)
        picked: list[dict[str, Any]] = []
        for score, start in ranked:
            end = min(video_duration, start + target_duration)
            if end - start < 4:
                continue
            if any(abs(start - item["start"]) < target_duration * 0.7 for item in picked):
                continue
            picked.append(
                {
                    "id": f"clip-{len(picked) + 1}",
                    "title": f"Highlight {len(picked) + 1}",
                    "reason": "Dipilih otomatis dari momen paling ramai",
                    "hook": "",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "score": min(1.0, 0.45 + score / (ranked[0][0] + 1e-6) * 0.5),
                    "selected": True,
                }
            )
            if len(picked) >= max_clips:
                break
        return picked
    except Exception:
        return []


def _take_best(clips: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    ranked = sorted(clips, key=lambda item: float(item.get("score") or 0), reverse=True)[:keep]
    for index, item in enumerate(ranked, start=1):
        item["id"] = f"clip-{index}"
        item["selected"] = True
    return ranked


def _parse_json_payload(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def suggest_clips(
    segments: list[dict],
    video_duration: float,
    target_duration: int,
    audio_path: str | None = None,
    clip_count: int = 3,
) -> list[dict[str, Any]]:
    if not segments:
        return fallback_clips(video_duration, target_duration, audio_path, clip_count)

    transcript = compact_transcript(segments)
    prompt = f"""Kamu editor clip otomatis. User TIDAK akan mengisi start/end.
Kamu HARUS memilih sendiri momen terbaik dari transkrip.

Durasi tiap clip HARUS {target_duration} detik (toleransi ±2 detik).
Durasi video: {video_duration:.1f} detik.
Jumlah clip yang HARUS kamu pilih: tepat {clip_count}.

Tugas:
- Pilih tepat {clip_count} momen PALING menarik (hook, punchline, emosi, insight padat).
- Tentukan start dan end sendiri, potong di jeda kalimat, jangan di tengah kata.
- Clip tidak boleh overlapping.
- Urutkan dari yang paling worth ditonton (score tertinggi dulu).
- title max 8 kata, hook 1 kalimat untuk overlay.

Kembalikan HANYA JSON valid:
{{"clips":[{{"title":"","reason":"","hook":"","start":0.0,"end":0.0,"score":0.8}}]}}

Transkrip:
{transcript}
"""

    for caller in _llm_callers():
        try:
            raw = caller(prompt)
            payload = _parse_json_payload(raw)
            items = payload.get("clips") if isinstance(payload, dict) else payload
            clips = _normalize_clips(items, segments, video_duration, target_duration)
            if clips:
                return _take_best(clips, clip_count)
        except Exception:
            continue
    return fallback_clips(video_duration, target_duration, audio_path, clip_count)


def _normalize_clips(
    items: Any,
    segments: list[dict],
    video_duration: float,
    target_duration: int,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    clips: list[dict[str, Any]] = []
    for index, item in enumerate(items[:8], start=1):
        if not isinstance(item, dict):
            continue
        start, end = snap_bounds(
            float(item.get("start") or 0),
            float(item.get("end") or 0),
            segments,
            video_duration,
            float(target_duration),
        )
        if end - start < 4:
            continue
        clips.append(
            {
                "id": f"clip-{index}",
                "title": str(item.get("title") or f"Clip {index}")[:80],
                "reason": str(item.get("reason") or "")[:200],
                "hook": str(item.get("hook") or item.get("title") or "")[:80],
                "start": start,
                "end": end,
                "score": max(0.0, min(1.0, float(item.get("score") or 0.6))),
                "selected": True,
            }
        )
    return clips


def _llm_callers():
    callers = []
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        callers.append(
            lambda prompt, key=deepseek_key, name=model: _chat_completions(
                "https://api.deepseek.com/chat/completions",
                key,
                name,
                prompt,
            )
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        callers.append(
            lambda prompt, key=openrouter_key, name=model: _chat_completions(
                "https://openrouter.ai/api/v1/chat/completions",
                key,
                name,
                prompt,
                extra_headers={
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "Cliper",
                },
            )
        )

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        callers.append(
            lambda prompt, key=openai_key: _chat_completions(
                "https://api.openai.com/v1/chat/completions",
                key,
                "gpt-4o-mini",
                prompt,
            )
        )
    return callers


def _chat_completions(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    extra_headers: dict[str, str] | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    response = httpx.post(
        url,
        headers=headers,
        json={
            "model": model,
            "temperature": 0.4,
            "messages": [
                {
                    "role": "system",
                    "content": "You return only valid JSON for video highlight clips.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
