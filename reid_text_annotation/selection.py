from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageStat

from .models import ImageRecord


@dataclass(frozen=True)
class ScoredImage:
    record: ImageRecord
    quality: float
    signature: tuple[float, ...]


def _score_image(record: ImageRecord) -> ScoredImage:
    with Image.open(record.path) as source:
        image = source.convert("RGB")
        width, height = image.size
        gray = image.convert("L")
        contrast = float(ImageStat.Stat(gray).stddev[0])
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = float(ImageStat.Stat(edges).stddev[0])
        signature_image = gray.resize((8, 8), Image.Resampling.BILINEAR)
        signature = tuple(value / 255.0 for value in signature_image.getdata())

    area_score = min(1.0, max(0.0, math.log2(max(width * height, 2)) / 18.0))
    contrast_score = min(1.0, contrast / 64.0)
    sharpness_score = min(1.0, sharpness / 64.0)
    quality = 0.45 * area_score + 0.25 * contrast_score + 0.30 * sharpness_score
    return ScoredImage(record=record, quality=quality, signature=signature)


def _signature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / max(1, len(left))


def select_images(records: list[ImageRecord], max_images: int) -> list[ImageRecord]:
    """Greedily balance readability with camera, split, and visual diversity."""
    if max_images <= 0:
        raise ValueError("max_images must be positive")

    scored: list[ScoredImage] = []
    for record in records:
        try:
            scored.append(_score_image(record))
        except (OSError, ValueError):
            continue
    if not scored:
        raise ValueError("No readable images remain in this group")

    scored.sort(key=lambda item: (-item.quality, str(item.record.path)))
    selected = [scored.pop(0)]
    used_cameras = {selected[0].record.camera_id}
    used_splits = {selected[0].record.split}

    while scored and len(selected) < max_images:
        def utility(item: ScoredImage) -> tuple[float, str]:
            visual_novelty = min(
                _signature_distance(item.signature, existing.signature)
                for existing in selected
            )
            new_camera = float(
                item.record.camera_id is not None
                and item.record.camera_id not in used_cameras
            )
            new_split = float(bool(item.record.split) and item.record.split not in used_splits)
            value = (
                0.55 * item.quality
                + 0.20 * new_camera
                + 0.10 * new_split
                + 0.15 * visual_novelty
            )
            return value, str(item.record.path)

        best = max(scored, key=utility)
        scored.remove(best)
        selected.append(best)
        used_cameras.add(best.record.camera_id)
        used_splits.add(best.record.split)

    return [item.record for item in selected]


def image_data_url(record: ImageRecord, max_side: int = 1024) -> str:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    with Image.open(record.path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
