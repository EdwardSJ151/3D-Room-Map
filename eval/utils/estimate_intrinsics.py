#!/usr/bin/env python3
"""Estimate pinhole intrinsics (fx, fy, cx, cy) for eval room images.

Usage:
    python eval/utils/estimate_intrinsics.py eval/images/room21.jpeg
    python eval/utils/estimate_intrinsics.py --batch eval/images --out eval/intrinsics
    python eval/utils/estimate_intrinsics.py --batch eval/images_upscale --out eval/intrinsics_upscale

Device auto-detection from filename:
    room21–23 → Samsung Galaxy Z Flip 7 (main rear camera)
    room24    → Meta Quest 3 passthrough (scaled from calibrated 1280² reference)

Upscaling: fx, fy, cx, cy scale with image width/height (same FOV, more pixels).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval"

DeviceId = Literal[
    "samsung_z_flip7_main",
    "samsung_z_flip7_ultrawide",
    "meta_quest3_passthrough",
    "fallback",
]


@dataclass(frozen=True)
class CameraPreset:
    name: str
    focal_mm: float | None = None
    sensor_w_mm: float | None = None
    sensor_h_mm: float | None = None
    hfov_deg: float | None = None
  # quest3 only — calibrated passthrough at 1280×1280
    ref_width: int | None = None
    ref_height: int | None = None
    ref_fx: float | None = None
    ref_fy: float | None = None
    ref_cx: float | None = None
    ref_cy: float | None = None


def sensor_4x3_mm(inch_fraction: float) -> tuple[float, float]:
    """Convert sensor inch fraction (e.g. 1/1.57) to 4:3 width/height in mm."""
    diagonal_mm = 16.0 / inch_fraction
    width_mm = diagonal_mm * 4.0 / 5.0
    height_mm = diagonal_mm * 3.0 / 5.0
    return width_mm, height_mm


def focal_mm_from_35mm_equiv(equiv_35mm: float, sensor_w_mm: float) -> float:
    return equiv_35mm * sensor_w_mm / 36.0


# Galaxy Z Flip 7 — main 50 MP, 1/1.57", ~24 mm equiv (typical Samsung wide)
_SW, _SH = sensor_4x3_mm(1.57)
_SAMSUNG_MAIN_F_MM = focal_mm_from_35mm_equiv(24.0, _SW)

# Ultrawide 12 MP, 1/3.2", 123° diagonal FOV (use horizontal ~103° for 4:3)
_UW_W, _UW_H = sensor_4x3_mm(3.2)

PRESETS: dict[DeviceId, CameraPreset] = {
    "samsung_z_flip7_main": CameraPreset(
        name="Samsung Galaxy Z Flip 7 — main (50 MP, 1/1.57\")",
        focal_mm=_SAMSUNG_MAIN_F_MM,
        sensor_w_mm=_SW,
        sensor_h_mm=_SH,
    ),
    "samsung_z_flip7_ultrawide": CameraPreset(
        name="Samsung Galaxy Z Flip 7 — ultrawide (12 MP, 1/3.2\", 123°)",
        hfov_deg=103.0,
    ),
    "meta_quest3_passthrough": CameraPreset(
        name="Meta Quest 3 — color passthrough (from headset calibration)",
        ref_width=1280,
        ref_height=1280,
        ref_fx=867.5797,
        ref_fy=867.5797,
        ref_cx=641.5017,
        ref_cy=637.3815,
    ),
    "fallback": CameraPreset(name="CuTR default fallback (no device info)"),
}


def read_exif_focal_mm(path: Path) -> float | None:
    try:
        exif = Image.open(path).getexif() or {}
    except Exception:
        return None
    raw = exif.get(37386)  # FocalLength
    if raw is None:
        return None
    if hasattr(raw, "numerator"):
        return float(raw.numerator) / float(raw.denominator)
    return float(raw)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return int(w), int(h)


def intrinsics_physical(
    focal_mm: float,
    sensor_w_mm: float,
    sensor_h_mm: float,
    width: int,
    height: int,
) -> dict[str, float | int]:
    fx = focal_mm / sensor_w_mm * width
    fy = focal_mm / sensor_h_mm * height
    return {
        "fx": round(fx, 6),
        "fy": round(fy, 6),
        "cx": round(width / 2.0, 6),
        "cy": round(height / 2.0, 6),
        "width": width,
        "height": height,
    }


def intrinsics_hfov(hfov_deg: float, width: int, height: int) -> dict[str, float | int]:
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx  # square pixels assumed
    return {
        "fx": round(fx, 6),
        "fy": round(fy, 6),
        "cx": round(width / 2.0, 6),
        "cy": round(height / 2.0, 6),
        "width": width,
        "height": height,
    }


def intrinsics_quest3_scaled(preset: CameraPreset, width: int, height: int) -> dict[str, float | int]:
    assert preset.ref_width and preset.ref_fx is not None
    sx = width / preset.ref_width
    sy = height / (preset.ref_height or preset.ref_width)
    return {
        "fx": round(preset.ref_fx * sx, 6),
        "fy": round((preset.ref_fy or preset.ref_fx) * sy, 6),
        "cx": round((preset.ref_cx or preset.ref_width / 2) * sx, 6),
        "cy": round((preset.ref_cy or preset.ref_height / 2) * sy, 6),
        "width": width,
        "height": height,
    }


def intrinsics_fallback(width: int, height: int) -> dict[str, float | int]:
    m = max(width, height)
    return {
        "fx": float(m),
        "fy": float(m),
        "cx": round(width / 2.0, 6),
        "cy": round(height / 2.0, 6),
        "width": width,
        "height": height,
    }


def room_number(path: Path) -> int | None:
    m = re.search(r"room(\d+)", path.stem, re.I)
    return int(m.group(1)) if m else None


def auto_device(path: Path) -> DeviceId:
    n = room_number(path)
    if n in (21, 22, 23):
        return "samsung_z_flip7_main"
    if n == 24:
        return "meta_quest3_passthrough"
    return "fallback"


def estimate(
    path: Path,
    device: DeviceId,
    prefer_exif: bool = True,
) -> dict:
    width, height = image_size(path)
    preset = PRESETS[device]
    method = device

    focal_mm = read_exif_focal_mm(path) if prefer_exif else None
    if focal_mm and preset.sensor_w_mm and preset.sensor_h_mm:
        intr = intrinsics_physical(focal_mm, preset.sensor_w_mm, preset.sensor_h_mm, width, height)
        method = f"{device}+exif_focal"
    elif device == "meta_quest3_passthrough":
        intr = intrinsics_quest3_scaled(preset, width, height)
    elif device == "samsung_z_flip7_ultrawide":
        intr = intrinsics_hfov(preset.hfov_deg or 103.0, width, height)
    elif device == "samsung_z_flip7_main" and preset.focal_mm and preset.sensor_w_mm:
        intr = intrinsics_physical(
            preset.focal_mm, preset.sensor_w_mm, preset.sensor_h_mm, width, height
        )
    elif device == "fallback":
        intr = intrinsics_fallback(width, height)
    else:
        intr = intrinsics_fallback(width, height)
        method = "fallback"

    return {
        **intr,
        "_source_image": str(path),
        "_device": device,
        "_method": method,
        "_preset": preset.name,
        "_note": (
            "Upscaled images: intrinsics scale with resolution; FOV unchanged. "
            "EXIF focal length overrides preset when present."
        ),
    }


def write_meta(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate camera intrinsics for eval images.")
    ap.add_argument("images", nargs="*", help="Image file(s)")
    ap.add_argument("--batch", type=Path, help="Process all images in directory")
    ap.add_argument("--out", type=Path, help="Output directory for meta JSON files")
    ap.add_argument(
        "--device",
        choices=list(PRESETS.keys()),
        help="Force device preset (default: auto from room number)",
    )
    ap.add_argument("--no-exif", action="store_true", help="Ignore EXIF focal length")
    ap.add_argument("--print-full", action="store_true", help="Include _metadata fields in stdout")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.batch:
        paths.extend(sorted(p for p in args.batch.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}))
    paths.extend(Path(p) for p in args.images)

    if not paths:
        ap.error("No images. Pass files or --batch <dir>.")

    out_dir = args.out

    for path in paths:
        if not path.is_file():
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        device = args.device or auto_device(path)
        result = estimate(path, device, prefer_exif=not args.no_exif)

        if out_dir:
            out_path = out_dir / f"{path.stem}.meta.json"
            write_meta(result, out_path)
            print(f"{path.name} → {out_path}  [{result['_method']}]")
        else:
            show = result if args.print_full else {k: v for k, v in result.items() if not k.startswith("_")}
            print(f"# {path.name}  device={device}  method={result['_method']}")
            print(json.dumps(show, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
