from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TEMPLATES: dict[str, dict[str, Any]] = {
    "viral": {
        "font": "Arial Black",
        "position": "center",
        "active_color": "#FFFF00",
        "idle_color": "#F0F0F0",
        "caption_size": "large",
        "crop": "tight",
        "quality": "good",
        "watermark": "",
        "zoom_punch": True,
        "hook": False,
        "fade": True,
        "safe_margin": 140,
        "blur_amount": 55,
    },
    "podcast": {
        "font": "Arial",
        "position": "bottom",
        "active_color": "#7CFFB2",
        "idle_color": "#E8E8E8",
        "caption_size": "normal",
        "crop": "loose",
        "quality": "good",
        "watermark": "",
        "zoom_punch": False,
        "hook": False,
        "fade": True,
        "safe_margin": 90,
        "blur_amount": 40,
    },
    "clean": {
        "font": "Arial",
        "position": "bottom",
        "active_color": "#FFFFFF",
        "idle_color": "#D0D0D0",
        "caption_size": "small",
        "crop": "normal",
        "quality": "fast",
        "watermark": "",
        "zoom_punch": False,
        "hook": False,
        "fade": True,
        "safe_margin": 80,
        "blur_amount": 30,
    },
}


@dataclass
class ClipStyle:
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
    blur_amount: int = 55

    @classmethod
    def from_payload(cls, data: dict[str, Any] | None) -> ClipStyle:
        raw = data or {}
        name = str(raw.get("template") or "viral")
        base = {**TEMPLATES.get(name, TEMPLATES["viral"]), **raw, "template": name}
        return cls(
            template=name,
            font=str(base.get("font") or "Arial Black"),
            position=str(base.get("position") or "center"),
            active_color=str(base.get("active_color") or "#FFFF00"),
            idle_color=str(base.get("idle_color") or "#F0F0F0"),
            caption_size=str(base.get("caption_size") or "normal"),
            crop=str(base.get("crop") or "normal"),
            quality=str(base.get("quality") or "fast"),
            watermark=str(base.get("watermark") or ""),
            zoom_punch=bool(base.get("zoom_punch", True)),
            hook=bool(base.get("hook", False)),
            fade=bool(base.get("fade", True)),
            safe_margin=int(base.get("safe_margin") or 120),
            blur_amount=max(0, min(100, int(base.get("blur_amount") or 55))),
        )

    def crop_factor(self) -> float:
        return {"tight": 2.1, "normal": 2.4, "loose": 3.0}.get(self.crop, 2.4)

    def blur_sigma(self) -> float:
        return round(2.0 + self.blur_amount * 0.42, 2)

    def encode(self) -> tuple[str, str]:
        if self.quality == "good":
            return "medium", "18"
        return "veryfast", "23"

    def font_px(self, orientation: str) -> tuple[int, int]:
        portrait = orientation == "portrait"
        sizes = {
            "small": (52, 40) if portrait else (42, 34),
            "normal": (68, 50) if portrait else (54, 42),
            "large": (82, 56) if portrait else (64, 46),
        }
        return sizes.get(self.caption_size, sizes["normal"])


def hex_to_ass(color: str) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    red, green, blue = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return f"&H00{blue:02X}{green:02X}{red:02X}&"
