from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image


def crop_bbox(
    image_path: Path,
    bbox_xyxy: Tuple[float, float, float, float],
    out_path: Path | None = None,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(round(float(x1))))
    y1 = max(0, int(round(float(y1))))
    x2 = min(w, int(round(float(x2))))
    y2 = min(h, int(round(float(y2))))
    crop = img.crop((x1, y1, x2, y2))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(str(out_path))
    return crop


def show_crop_cli(crop: Image.Image, label: str = "") -> None:
    try:
        crop.show(title=label)
    except Exception:
        print(f"[image_utils] cannot display image (no GUI): {label}")
