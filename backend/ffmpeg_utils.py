from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .style import ClipStyle, hex_to_ass


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe tidak ditemukan. Install ffmpeg dan pastikan ada di PATH."
        )


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "ffmpeg gagal").strip()
        raise RuntimeError(err[-2000:])
    return result


def probe_duration(path: str | Path) -> float:
    require_ffmpeg()
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout or "{}")
    return float(data.get("format", {}).get("duration") or 0)


def extract_audio(video_path: str | Path, audio_path: str | Path) -> Path:
    require_ffmpeg()
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )
    return audio_path


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _ass_text(text: str) -> str:
    return (
        re.sub(r"[\r\n]+", " ", text)
        .strip()
        .replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _collect_words(segments: list[dict], clip_start: float, clip_end: float) -> list[dict]:
    words: list[dict] = []
    for seg in segments:
        raw_words = seg.get("words") or []
        if raw_words:
            for word in raw_words:
                token = str(word.get("word") or "").strip()
                start = float(word.get("start") or 0)
                end = float(word.get("end") or 0)
                if not token or end <= clip_start or start >= clip_end:
                    continue
                words.append(
                    {
                        "word": token,
                        "start": max(0.0, start - clip_start),
                        "end": min(clip_end - clip_start, end - clip_start),
                    }
                )
            continue
        text = str(seg.get("text") or "").strip()
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or 0)
        if not text or end <= clip_start or start >= clip_end:
            continue
        tokens = text.split()
        rel_start = max(0.0, start - clip_start)
        rel_end = min(clip_end - clip_start, end - clip_start)
        span = max(0.2, rel_end - rel_start)
        step = span / max(1, len(tokens))
        for index, token in enumerate(tokens):
            words.append(
                {
                    "word": token,
                    "start": rel_start + index * step,
                    "end": rel_start + (index + 1) * step,
                }
            )
    return words


def _chunk_words(words: list[dict], max_words: int = 4) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for word in words:
        current.append(word)
        chars += len(word["word"])
        if len(current) >= max_words or chars >= 26:
            chunks.append(current)
            current = []
            chars = 0
    if current:
        chunks.append(current)
    return chunks


def _karaoke_line(
    chunk: list[dict],
    active: int,
    *,
    active_color: str,
    idle_color: str,
    alignment: int,
) -> str:
    parts: list[str] = []
    for index, word in enumerate(chunk):
        token = _ass_text(word["word"])
        if index == active:
            parts.append(rf"{{\c{active_color}\b1}}" + token + rf"{{\c{idle_color}\b0}}")
        else:
            parts.append(rf"{{\c{idle_color}}}" + token)
    return rf"{{\an{alignment}\q2}}" + " ".join(parts)


def write_ass_overlay(
    dest: Path,
    clip_start: float,
    clip_end: float,
    *,
    segments: list[dict] | None = None,
    hook: str = "",
    caption: bool = True,
    hook_overlay: bool = True,
    orientation: str = "landscape",
    style: ClipStyle | None = None,
) -> Path | None:
    look = style or ClipStyle()
    events: list[str] = []
    if hook_overlay and hook.strip():
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(2.4)},Hook,,0,0,0,,{_ass_text(hook[:70])}"
        )

    active = hex_to_ass(look.active_color)
    idle = hex_to_ass(look.idle_color)
    alignment = 5 if look.position == "center" else 2
    if caption and segments:
        for chunk in _chunk_words(_collect_words(segments, clip_start, clip_end)):
            for index, word in enumerate(chunk):
                start = word["start"]
                end = max(word["end"], start + 0.12)
                line = _karaoke_line(
                    chunk,
                    index,
                    active_color=active,
                    idle_color=idle,
                    alignment=alignment,
                )
                events.append(
                    f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{line}"
                )

    duration = max(0.4, clip_end - clip_start)
    if caption and not any("Caption" in item for item in events) and hook.strip():
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(min(duration, 3.2))},Caption,,0,0,0,,{_ass_text(hook[:80])}"
        )
    mark = look.watermark.strip()
    if mark:
        events.append(
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(duration)},Mark,,0,0,0,,{_ass_text(mark[:40])}"
        )

    if not events:
        return None

    play_x, play_y = (1080, 1920) if orientation == "portrait" else (1920, 1080)
    caption_size, hook_size = look.font_px(orientation)
    font = look.font.replace(",", " ")
    margin = max(40, look.safe_margin)
    dest.write_text(
        "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {play_x}",
                f"PlayResY: {play_y}",
                "WrapStyle: 0",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                f"Style: Caption,{font},{caption_size},{idle},&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,0,{alignment},80,80,{margin},1",
                f"Style: Hook,{font},{hook_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,8,70,70,{max(70, margin - 20)},1",
                f"Style: Mark,{font},36,&H00E8E8E8,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,0,3,48,48,52,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
                *events,
            ]
        ),
        encoding="utf-8",
    )
    return dest


def _ffmpeg_sub_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")


def _full_cover_filters(
    video_path: str | Path,
    start: float,
    end: float,
    out_w: int,
    out_h: int,
    *,
    orientation: str,
    focus_speaker: bool,
    crop_factor: float,
    audio_path: str | None,
    cmd_path: Path,
) -> list[str]:
    if focus_speaker:
        from .tracker import track_speaker, write_sendcmd

        track = track_speaker(
            str(video_path),
            start,
            end,
            orientation,
            audio_path=audio_path,
            crop_factor=crop_factor,
        )
        if track:
            box = track.primary_box()
            moved = any(item[1] != box.x or item[2] != box.y for item in track.keyframes)
            if moved:
                write_sendcmd(track, cmd_path)
                return [
                    f"sendcmd=f='{_ffmpeg_sub_path(cmd_path)}'",
                    f"crop={box.w}:{box.h}:{box.x}:{box.y}:exact=0",
                    f"scale={out_w}:{out_h}:flags=lanczos",
                ]
            return [
                f"crop={box.w}:{box.h}:{box.x}:{box.y}",
                f"scale={out_w}:{out_h}:flags=lanczos",
            ]
    return [
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase",
        f"crop={out_w}:{out_h}",
    ]


def export_soft_clip(
    video_path: str | Path,
    dest: Path,
    start: float,
    end: float,
    *,
    hook: str = "",
    segments: list[dict] | None = None,
    fade: bool = True,
    caption: bool = True,
    hook_overlay: bool = True,
    orientation: str = "landscape",
    focus_speaker: bool = True,
    style: ClipStyle | None = None,
    audio_path: str | None = None,
    portrait_mode: str = "full",
) -> Path:
    require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    look = style or ClipStyle()
    duration = max(0.4, end - start)
    fade_in = 0.4 if fade else 0.0
    fade_out = 0.5 if fade else 0.0
    if duration < 1.2:
        fade_in = min(fade_in, 0.2)
        fade_out = min(fade_out, 0.25)

    if orientation == "portrait":
        out_w, out_h = 1080, 1920
    else:
        out_w, out_h = 1920, 1080

    mode = portrait_mode if orientation == "portrait" else "full"
    cmd_path = dest.with_suffix(".crop.txt")
    finish: list[str] = []
    if look.zoom_punch and mode == "full":
        finish.append("scale='iw*(1+0.08*min(t/1.6\\,1))':'ih*(1+0.08*min(t/1.6\\,1))':eval=frame")
        finish.append(f"crop={out_w}:{out_h}:(in_w-{out_w})/2:(in_h-{out_h})*0.32")
    if fade:
        fade_out_start = max(0.0, duration - fade_out)
        finish.append(f"fade=t=in:st=0:d={fade_in}")
        finish.append(f"fade=t=out:st={fade_out_start}:d={fade_out}")

    ass_path = dest.with_suffix(".ass")
    written = write_ass_overlay(
        ass_path,
        start,
        end,
        segments=segments,
        hook=hook,
        caption=caption,
        hook_overlay=hook_overlay,
        orientation=orientation,
        style=look,
    )
    if written:
        finish.append(f"subtitles='{_ffmpeg_sub_path(written)}'")
    finish.append("format=yuv420p")
    finish.append("setsar=1")

    use_complex = mode == "blur"
    if mode == "letterbox":
        vf = [
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos",
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black",
            *finish,
        ]
        graph = ",".join(vf)
    elif mode == "blur":
        sigma = min(18.0, max(2.0, look.blur_sigma() * 0.45))
        graph = (
            f"[0:v]split=2[bgin][fgin];"
            f"[bgin]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,gblur=sigma={sigma},"
            f"scale={out_w}:{out_h}:flags=bilinear,setsar=1[bg];"
            f"[fgin]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto,{','.join(finish)}[vout]"
        )
    else:
        vf = [
            *_full_cover_filters(
                video_path,
                start,
                end,
                out_w,
                out_h,
                orientation=orientation,
                focus_speaker=focus_speaker,
                crop_factor=look.crop_factor(),
                audio_path=audio_path,
                cmd_path=cmd_path,
            ),
            *finish,
        ]
        graph = ",".join(vf)

    af: list[str] = []
    if fade:
        fade_out_start = max(0.0, duration - fade_out)
        af.append(f"afade=t=in:st=0:d={fade_in}")
        af.append(f"afade=t=out:st={fade_out_start}:d={fade_out}")
    af.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    preset, crf = look.encode()
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-filter_complex" if use_complex else "-vf",
        graph,
    ]
    if use_complex:
        cmd.extend(["-map", "[vout]", "-map", "0:a?"])
    cmd.extend(
        [
            "-af",
            ",".join(af),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    try:
        run(cmd)
    finally:
        if ass_path.exists():
            ass_path.unlink(missing_ok=True)
        if cmd_path.exists():
            cmd_path.unlink(missing_ok=True)
    return dest
