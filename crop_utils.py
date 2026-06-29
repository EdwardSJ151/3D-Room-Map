from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def crop_for_detection(img: Image.Image, det: Dict[str, Any]) -> Optional[Image.Image]:
    bbox = det.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = _clamp(x1, 0.0, w - 1.0)
    y1 = _clamp(y1, 0.0, h - 1.0)
    x2 = _clamp(x2, 0.0, w - 1.0)
    y2 = _clamp(y2, 0.0, h - 1.0)
    x1_i = max(0, int(round(x1)))
    y1_i = max(0, int(round(y1)))
    x2_i = max(0, int(round(x2)))
    y2_i = max(0, int(round(y2)))
    if x2_i - x1_i < 2 or y2_i - y1_i < 2:
        return None
    return img.crop((x1_i, y1_i, x2_i, y2_i))


def save_detection_crops(
    img: Image.Image,
    detections: List[Dict[str, Any]],
    out_dir: Path,
    *,
    quality: int = 90,
) -> List[Tuple[int, Image.Image]]:
    """Save crops as {out_dir}/{idx}.jpg. Returns (idx, crop) pairs saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    crops: List[Tuple[int, Image.Image]] = []
    for i, det in enumerate(detections):
        crop = crop_for_detection(img, det)
        if crop is None:
            continue
        crop.convert("RGB").save(out_dir / f"{i}.jpg", format="JPEG", quality=quality)
        crops.append((i, crop))
    return crops
