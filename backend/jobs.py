from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class Job:
    id: str
    status: str = "idle"
    step: str = "ready"
    error: str | None = None
    title: str = ""
    duration: float = 0.0
    video_path: str = ""
    audio_path: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    clips: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, str]] = field(default_factory=list)
    last_target_duration: int = 0
    last_clip_count: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "step": self.step,
            "error": self.error,
            "title": self.title,
            "duration": self.duration,
            "clips": self.clips,
            "exports": self.exports,
            "video_url": f"/api/media/{self.id}" if self.video_path else None,
            "lines": [
                {
                    "start": float(seg.get("start") or 0),
                    "end": float(seg.get("end") or 0),
                    "text": str(seg.get("text") or "").strip(),
                }
                for seg in self.segments
                if str(seg.get("text") or "").strip()
            ],
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, **kwargs: Any) -> Job:
        job = Job(id=uuid4().hex[:12], **kwargs)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def require(self, job_id: str) -> Job:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job


store = JobStore()
