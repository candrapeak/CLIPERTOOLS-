from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

_YUNET = None
_YUNET_LOCK = Lock()
_CASCADES: dict[str, list[cv2.CascadeClassifier]] | None = None


@dataclass
class CropBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceTrack:
    frame_w: int
    frame_h: int
    crop_w: int
    crop_h: int
    keyframes: list[tuple[float, int, int, int, int]]

    def static_box(self) -> CropBox:
        if not self.keyframes:
            return CropBox(0, 0, self.crop_w, self.crop_h)
        xs = np.array([item[1] for item in self.keyframes], dtype=np.float32)
        ys = np.array([item[2] for item in self.keyframes], dtype=np.float32)
        return CropBox(_even(float(np.median(xs))), _even(float(np.median(ys))), self.crop_w, self.crop_h)

    def primary_box(self) -> CropBox:
        if not self.keyframes:
            return self.static_box()
        _t, x, y, w, h = self.keyframes[0]
        return CropBox(x, y, w, h)

    def focus_points(self) -> list[tuple[float, float]]:
        width = max(self.frame_w, 1)
        return [(t, (x + w / 2) / width) for t, x, _y, w, _h in self.keyframes]


def _even(value: float) -> int:
    number = max(2, int(round(value)))
    return number - (number % 2)


def _yunet() -> cv2.FaceDetectorYN | None:
    global _YUNET
    if _YUNET is not None:
        return _YUNET
    model = Path(__file__).resolve().parent / "models" / "face_yunet.onnx"
    if not model.exists():
        return None
    try:
        _YUNET = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.36, 0.3, 5000)
        return _YUNET
    except cv2.error:
        return None


def _cascades() -> dict[str, list[cv2.CascadeClassifier]]:
    global _CASCADES
    if _CASCADES is not None:
        return _CASCADES
    root = getattr(cv2.data, "haarcascades", "")
    loaded: dict[str, list[cv2.CascadeClassifier]] = {"face": [], "profile": []}
    for name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ):
        cascade = cv2.CascadeClassifier(root + name)
        if not cascade.empty():
            loaded["face"].append(cascade)
            if "profile" in name:
                loaded["profile"].append(cascade)
    _CASCADES = loaded
    return loaded


def _nms(boxes: list[tuple[int, int, int, int, float]], overlap: float = 0.35) -> list[tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    array = np.array([item[:4] for item in boxes], dtype=np.float32)
    scores = np.array([item[4] for item in boxes], dtype=np.float32)
    x1, y1 = array[:, 0], array[:, 1]
    x2, y2 = x1 + array[:, 2], y1 + array[:, 3]
    area = np.maximum(1.0, array[:, 2] * array[:, 3])
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[index] + area[order[1:]] - inter + 1e-6)
        order = order[1:][iou < overlap]
    return [boxes[i] for i in keep]


def _focus_from_box(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    return x + w / 2, y + h * 0.36


def _detect_yunet(frame: np.ndarray) -> list[tuple[int, int, int, int, float, float, float]]:
    detector = _yunet()
    if detector is None:
        return []
    height, width = frame.shape[:2]
    try:
        with _YUNET_LOCK:
            detector.setInputSize((width, height))
            _ok, hits = detector.detect(frame)
    except cv2.error:
        return []
    if hits is None or len(hits) == 0:
        return []
    out: list[tuple[int, int, int, int, float, float, float]] = []
    for row in hits:
        x, y, w, h = [int(v) for v in row[:4]]
        score = float(row[-1]) if len(row) > 4 else 0.7
        if w < 12 or h < 12:
            continue
        bx, by = _focus_from_box(x, y, w, h)
        if len(row) >= 10:
            nx, ny = float(row[8]), float(row[9])
            fx = nx * 0.74 + bx * 0.26
            fy = ny * 0.50 + by * 0.50
        else:
            fx, fy = bx, by
        out.append((max(0, x), max(0, y), w, h, score, fx, fy))
    return out


def _detect_haar(gray: np.ndarray, min_size: int) -> list[tuple[int, int, int, int, float, float, float]]:
    found: list[tuple[int, int, int, int, float, float, float]] = []
    packs = _cascades()
    for cascade in packs["face"]:
        hits = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in hits:
            ix, iy, iw, ih = int(x), int(y), int(w), int(h)
            fx, fy = _focus_from_box(ix, iy, iw, ih)
            found.append((ix, iy, iw, ih, 0.55, fx, fy))
    width = gray.shape[1]
    flipped = cv2.flip(gray, 1)
    for cascade in packs["profile"]:
        hits = cascade.detectMultiScale(
            flipped,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in hits:
            ix, iy, iw, ih = width - int(x) - int(w), int(y), int(w), int(h)
            fx, fy = _focus_from_box(ix, iy, iw, ih)
            found.append((ix, iy, iw, ih, 0.5, fx, fy))
    return _nms(found)


def _plausible_face(box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
    x, y, w, h = box
    if w < 14 or h < 14:
        return False
    if y > frame_h * 0.72:
        return False
    ratio = h / max(w, 1)
    if ratio < 0.65 or ratio > 1.85:
        return False
    if w < frame_w * 0.035 and h < frame_h * 0.06:
        return False
    # tiny object sitting on a table in the dead center
    cx = x + w / 2
    if w < frame_w * 0.08 and y > frame_h * 0.5 and 0.38 * frame_w < cx < 0.62 * frame_w:
        return False
    return True


def _mouth_score(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    my = y + int(h * 0.55)
    mh = max(4, int(h * 0.4))
    mx = x + int(w * 0.16)
    mw = max(4, int(w * 0.68))
    roi = gray[my : my + mh, mx : mx + mw]
    if roi.size < 20:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _face_score(
    box: tuple[int, int, int, int],
    score: float,
    gray: np.ndarray,
    frame_w: int,
    frame_h: int,
    voiced: float,
) -> float:
    x, y, w, h = box
    area = (w * h) / max(1.0, frame_w * frame_h)
    cx = (x + w / 2) / max(frame_w, 1)
    center = 1.0 - min(1.0, abs(cx - 0.5) * 1.4)
    mouth = _mouth_score(gray, x, y, w, h)
    return score * 90 + area * 820 + center * 18 + mouth * 0.28 + voiced * 32


def _audio_envelope(audio_path: str | None, hop: float = 0.05) -> tuple[np.ndarray, float]:
    if not audio_path or not Path(audio_path).exists():
        return np.zeros(0, dtype=np.float32), hop
    try:
        with wave.open(audio_path, "rb") as handle:
            rate = handle.getframerate() or 16000
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels() or 1
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        window = max(1, int(rate * hop))
        usable = samples[: len(samples) - (len(samples) % window)]
        if usable.size == 0:
            return np.zeros(0, dtype=np.float32), hop
        chunks = usable.reshape(-1, window)
        rms = np.sqrt(np.mean(chunks * chunks, axis=1))
        peak = float(np.percentile(rms, 90)) or 1.0
        return (rms / peak).astype(np.float32), hop
    except Exception:
        return np.zeros(0, dtype=np.float32), hop


def _voiced(envelope: np.ndarray, hop: float, t: float) -> float:
    if envelope.size == 0:
        return 0.55
    lo = max(0, int((t - 0.12) / hop))
    hi = min(envelope.size, int((t + 0.18) / hop) + 1)
    if lo >= hi:
        return 0.0
    return float(np.max(envelope[lo:hi]))


def crop_from_face(
    face: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    orientation: str,
    crop_factor: float = 2.4,
    kind: str = "face",
) -> CropBox:
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    cy = fy + fh * (0.36 if kind == "face" else 0.22)
    return _window_at(cx, cy, frame_w, frame_h, orientation, crop_factor, fh)


def _window_at(
    cx: float,
    cy: float,
    frame_w: int,
    frame_h: int,
    orientation: str,
    crop_factor: float,
    face_h: float = 0.0,
) -> CropBox:
    aspect_w, aspect_h = (9, 16) if orientation == "portrait" else (16, 9)
    face_h = max(float(face_h), frame_h * 0.12)
    if orientation == "portrait":
        tall = 2.7 + crop_factor * 0.58
        crop_h = min(float(frame_h), max(face_h * 3.15, face_h * tall))
        crop_w = crop_h * aspect_w / aspect_h
        if crop_w > frame_w:
            crop_w = float(frame_w)
            crop_h = crop_w * aspect_h / aspect_w
        if crop_h > frame_h:
            crop_h = float(frame_h)
            crop_w = crop_h * aspect_w / aspect_h
        crop_w = max(280.0, crop_w)
        crop_h = min(float(frame_h), crop_w * aspect_h / aspect_w)
    else:
        crop_h = min(float(frame_h), max(frame_h * 0.62, face_h * (2.8 + crop_factor * 0.4)))
        crop_w = crop_h * aspect_w / aspect_h
        if crop_w > frame_w:
            crop_w = float(frame_w)
            crop_h = crop_w * aspect_h / aspect_w
        if crop_h > frame_h:
            crop_h = float(frame_h)
            crop_w = crop_h * aspect_w / aspect_h
    x = min(max(0.0, cx - crop_w / 2), frame_w - crop_w)
    y = min(max(0.0, cy - crop_h * 0.30), frame_h - crop_h)
    return CropBox(_even(x), _even(y), _even(crop_w), _even(crop_h))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, aw * ah + bw * bh - inter)
    return inter / union


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] * 0.38


def _area(box: tuple[int, int, int, int]) -> float:
    return float(max(1, box[2] * box[3]))


def _collect_faces(small: np.ndarray, gray: np.ndarray, min_face: int) -> list:
    hits = _detect_yunet(small)
    need_more = not hits or (len(hits) == 1 and hits[0][2] < small.shape[1] * 0.095)
    if not hits:
        lifted = cv2.convertScaleAbs(small, alpha=1.28, beta=16)
        hits = _detect_yunet(lifted)
        need_more = not hits or need_more
    if need_more:
        hits = _nms(list(hits) + _detect_haar(gray, min_face))
    return hits


def _stable_size(
    raw: list[tuple[float, float, float, float, float]],
    frame_w: int,
    frame_h: int,
    orientation: str,
) -> tuple[int, int]:
    widths = np.array([item[3] for item in raw], dtype=np.float32)
    crop_w = float(np.percentile(widths, 72))
    if orientation == "portrait":
        crop_h = crop_w * 16 / 9
    else:
        crop_h = crop_w * 9 / 16
    if crop_h > frame_h:
        crop_h = float(frame_h)
        crop_w = crop_h * (9 / 16 if orientation == "portrait" else 16 / 9)
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = crop_w * (16 / 9 if orientation == "portrait" else 9 / 16)
    return _even(crop_w), _even(crop_h)


def _box_at(cx: float, cy: float, frame_w: int, frame_h: int, crop_w: int, crop_h: int) -> tuple[int, int]:
    x = min(max(0.0, cx - crop_w / 2), frame_w - crop_w)
    y = min(max(0.0, cy - crop_h * 0.30), frame_h - crop_h)
    return int(round(x)), int(round(y))


def _locked_shots(
    focuses: list[tuple[float, float, float]],
    duration: float,
    frame_w: int,
    frame_h: int,
    crop_w: int,
    crop_h: int,
) -> list[tuple[float, int, int, int, int]]:
    if not focuses:
        return []
    jump = frame_w * 0.16
    segments: list[tuple[float, list[float], list[float]]] = [(0.0, [focuses[0][1]], [focuses[0][2]])]
    pending = 0
    for stamp, cx, cy in focuses:
        lock_x = float(np.median(segments[-1][1]))
        if abs(cx - lock_x) >= jump:
            pending += 1
            if pending >= 3:
                segments.append((stamp, [cx], [cy]))
                pending = 0
            continue
        pending = 0
        segments[-1][1].append(cx)
        segments[-1][2].append(cy)

    keyframes: list[tuple[float, int, int, int, int]] = []
    for stamp, xs, ys in segments:
        x, y = _box_at(float(np.median(xs)), float(np.median(ys)), frame_w, frame_h, crop_w, crop_h)
        point = (round(max(0.0, stamp), 3), x, y, crop_w, crop_h)
        if not keyframes or point[1] != keyframes[-1][1] or point[2] != keyframes[-1][2]:
            keyframes.append(point)
    if not keyframes:
        return []
    keyframes[0] = (0.0, *keyframes[0][1:])
    if keyframes[-1][0] < duration - 0.03:
        keyframes.append((round(duration, 3), *keyframes[-1][1:]))
    return keyframes


def _classify_faces(
    faces: list[tuple[int, int, int, int, float]],
    frame_w: int,
    frame_h: int,
) -> tuple[str, list[tuple[int, int, int, int, float]]]:
    if not faces:
        return "none", []
    ranked = sorted(faces, key=lambda item: item[2] * item[3] * (0.35 + item[4]), reverse=True)
    lead = ranked[0]
    lead_area = _area(lead[:4])
    frame_area = max(1.0, frame_w * frame_h)
    closeup = lead_area > frame_area * 0.016 or lead[2] > frame_w * 0.10
    if closeup and (len(ranked) == 1 or _area(ranked[1][:4]) < lead_area * 0.48):
        return "closeup", [lead]
    wide: list[tuple[int, int, int, int, float]] = []
    for item in ranked[:4]:
        if _area(item[:4]) < lead_area * 0.28:
            continue
        wide.append(item)
    if len(wide) >= 2:
        left = min(wide, key=lambda item: item[0] + item[2] / 2)
        right = max(wide, key=lambda item: item[0] + item[2] / 2)
        gap = (right[0] + right[2] / 2) - (left[0] + left[2] / 2)
        if gap > frame_w * 0.20:
            return "dual", [left, right]
    return "single", [lead]


def track_speaker(
    video_path: str,
    start: float,
    end: float,
    orientation: str = "portrait",
    audio_path: str | None = None,
    crop_factor: float = 2.4,
    sample_step: float = 0.08,
) -> FaceTrack | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width < 32 or height < 32:
        cap.release()
        return None

    envelope, hop = _audio_envelope(audio_path)
    scale = min(1.0, 960.0 / width)
    sw, sh = max(2, int(width * scale)), max(2, int(height * scale))
    inv = width / sw
    min_face = max(18, min(sw, sh) // 26)
    samples: list[tuple[float, str, list[tuple[int, int, int, int, float]], np.ndarray, float]] = []
    last_small: tuple | None = None
    last_t = -10.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000)
    next_t = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
        if t < start - 0.05:
            continue
        if t > end:
            break
        if t + 0.001 < next_t:
            continue
        next_t = t + sample_step
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        voiced = _voiced(envelope, hop, t)
        hits = _collect_faces(small, gray, min_face)
        faces = [item for item in hits if _plausible_face(item[:4], sw, sh)]
        if not faces and last_small is not None:
            x, y, w, h = last_small[:4]
            pad = int(max(w, h) * 0.9)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            roi = small[y0 : min(sh, y + h + pad), x0 : min(sw, x + w + pad)]
            if roi.size > 80:
                local = _detect_yunet(roi)
                for hit in local:
                    lx, ly, lw, lh, sc = hit[:5]
                    fx = (hit[5] + x0) if len(hit) > 5 else lx + x0 + lw / 2
                    fy = (hit[6] + y0) if len(hit) > 6 else ly + y0 + lh * 0.36
                    faces.append((lx + x0, ly + y0, lw, lh, sc, fx, fy))
                faces = [item for item in faces if _plausible_face(item[:4], sw, sh)]
        kind, picked = _classify_faces(faces, sw, sh)
        if picked:
            last_small = picked[0]
            last_t = t
        elif last_small is not None and (t - last_t) <= 0.75:
            kind, picked = "hold", [last_small]
        if not picked:
            continue
        samples.append((t, kind, picked, gray, voiced))
    cap.release()
    if not samples:
        return None

    last_box: tuple[int, int, int, int] | None = None
    dual_side: int | None = None
    pending_side: int | None = None
    pending_hits = 0
    hold_cx = hold_cy = None
    raw: list[tuple[float, float, float, float, float]] = []

    for stamp, kind, faces, gray, voiced in samples:
        scored = [
            (
                item,
                _face_score(item[:4], item[4], gray, sw, sh, voiced)
                + (_iou(item[:4], last_box) * 62 if last_box else 0),
            )
            for item in faces
        ]
        chosen = max(scored, key=lambda item: item[1])[0]
        if kind == "dual" and len(faces) >= 2:
            left, right = faces[0], faces[1]
            if left[0] + left[2] / 2 > right[0] + right[2] / 2:
                left, right = right, left
            pair = {0: left, 1: right}
            left_s = _face_score(left[:4], 0.8, gray, sw, sh, voiced)
            right_s = _face_score(right[:4], 0.8, gray, sw, sh, voiced)
            winner = 0 if left_s >= right_s else 1
            if dual_side is None:
                dual_side = winner
            elif winner != dual_side:
                if pending_side == winner:
                    pending_hits += 1
                else:
                    pending_side = winner
                    pending_hits = 1
                if pending_hits >= 5:
                    dual_side = winner
                    pending_side = None
                    pending_hits = 0
            else:
                pending_side = None
                pending_hits = 0
            chosen = pair[dual_side]
        else:
            dual_side = None
            pending_side = None
            pending_hits = 0

        target = chosen[:4]
        full = (
            int(target[0] * inv),
            int(target[1] * inv),
            max(2, int(target[2] * inv)),
            max(2, int(target[3] * inv)),
        )
        if len(chosen) >= 7:
            cx, cy = float(chosen[5]) * inv, float(chosen[6]) * inv
        else:
            cx, cy = _center(full)
        if hold_cx is not None and abs(cx - hold_cx) < 18 and abs(cy - hold_cy) < 14:
            cx, cy = hold_cx, hold_cy
        else:
            hold_cx, hold_cy = cx, cy
        last_box = target
        win = _window_at(cx, cy, width, height, orientation, crop_factor, full[3])
        raw.append((round(max(0.0, stamp - start), 3), cx, cy, float(win.w), float(win.h)))

    if not raw:
        return None
    raw[0] = (0.0, *raw[0][1:])
    duration = max(0.4, end - start)
    if raw[-1][0] < duration - 0.04:
        raw.append((round(duration, 3), *raw[-1][1:]))
    crop_w, crop_h = _stable_size(raw, width, height, orientation)
    focuses = [(item[0], item[1], item[2]) for item in raw]
    keyframes = _locked_shots(focuses, duration, width, height, crop_w, crop_h)
    if not keyframes:
        return None
    return FaceTrack(width, height, crop_w, crop_h, keyframes)


def speaker_crop(
    video_path: str,
    start: float,
    end: float,
    orientation: str = "portrait",
    audio_path: str | None = None,
    crop_factor: float = 2.4,
) -> CropBox | None:
    track = track_speaker(video_path, start, end, orientation, audio_path, crop_factor)
    return track.static_box() if track else None


def write_sendcmd(track: FaceTrack, dest: Path) -> Path:
    lines = [f"{t:.4f} crop x {x}, crop y {y};" for t, x, y, _w, _h in track.keyframes]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
