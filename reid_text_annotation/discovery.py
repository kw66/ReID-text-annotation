from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import AnnotationGroup, ImageRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PUBLIC_RGB_DATASETS = {"market1501", "dukemtmc", "cuhk03", "msmt17"}
_PID_CAMERA_PATTERN = re.compile(r"^(-?\d+)_c(\d+)", re.IGNORECASE)
_ATUSTC_PATTERN = re.compile(
    r"p(\d+)-d(\d+)-c(\d+)[\\/]cam(\d+)-f(\d+)-(\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def _image_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return (
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _resolve_public_root(data_root: Path, dataset: str) -> Path:
    root = data_root.resolve()
    nested_names = {
        "market1501": ["Market-1501-v15.09.15"],
        "dukemtmc": ["DukeMTMC-reID"],
        "cuhk03": ["detected"],
        "msmt17": ["MSMT17_V1", "MSMT17_V2"],
    }
    if (root / "bounding_box_train").exists():
        return root
    for name in nested_names[dataset]:
        candidate = root / name
        if (candidate / "bounding_box_train").exists():
            return candidate
    return root


def _parse_public_name(path: Path, dataset: str) -> tuple[int, int | None] | None:
    match = _PID_CAMERA_PATTERN.match(path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))

    parts = path.stem.split("_")
    if not parts:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None

    camera_id = None
    if dataset in {"cuhk03", "msmt17"} and len(parts) > 1:
        camera_token = parts[1].lstrip("cC")
        if camera_token.isdigit():
            camera_id = int(camera_token)
    return pid, camera_id


def discover_public_rgb(data_root: Path, dataset: str) -> list[AnnotationGroup]:
    dataset = dataset.strip().lower()
    if dataset not in PUBLIC_RGB_DATASETS:
        raise ValueError(f"Unsupported public RGB dataset: {dataset}")

    root = _resolve_public_root(data_root, dataset)
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    split_dirs = (
        ("train", root / "bounding_box_train"),
        ("query", root / "query"),
        ("gallery", root / "bounding_box_test"),
    )
    for split, directory in split_dirs:
        for path in _image_files(directory):
            parsed = _parse_public_name(path, dataset)
            if parsed is None:
                continue
            pid, camera_id = parsed
            if (dataset == "msmt17" and pid < 0) or (dataset != "msmt17" and pid <= 0):
                continue
            grouped[pid].append(
                ImageRecord(path=path.resolve(), camera_id=camera_id, split=split)
            )

    if not grouped:
        raise ValueError(
            "No valid images were discovered. Check the dataset root and expected "
            "bounding_box_train/query/bounding_box_test layout."
        )
    return [
        AnnotationGroup(
            dataset=dataset,
            protocol="public-rgb",
            pid=pid,
            images=sorted(records, key=lambda item: str(item.path)),
        )
        for pid, records in sorted(grouped.items())
    ]


def discover_atustc(data_root: Path) -> list[AnnotationGroup]:
    root = data_root.resolve()
    grouped: dict[tuple[int, int, str], list[ImageRecord]] = defaultdict(list)
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        match = _ATUSTC_PATTERN.search(relative)
        if match is None:
            continue
        pid, _day, clothes_id, camera_id, _slot, frame_index, _suffix = match.groups()
        camera = int(camera_id)
        modality = "rgb" if camera <= 8 else "ir"
        grouped[(int(pid), int(clothes_id), modality)].append(
            ImageRecord(
                path=path.resolve(),
                camera_id=camera,
                split="all",
                frame_index=int(frame_index),
            )
        )

    if not grouped:
        raise ValueError(
            "No valid RGB/IR images were discovered. Use --dataset manifest for a "
            "different directory or filename convention."
        )
    return [
        AnnotationGroup(
            dataset="atustc",
            protocol="rgb-ir",
            pid=pid,
            clothes_id=clothes_id,
            modality=modality,
            images=sorted(
                records,
                key=lambda item: (
                    item.camera_id if item.camera_id is not None else -1,
                    item.frame_index,
                    str(item.path),
                ),
            ),
        )
        for (pid, clothes_id, modality), records in sorted(grouped.items())
    ]


def _manifest_image(value: object, base_dir: Path) -> ImageRecord:
    if isinstance(value, str):
        raw_path = value
        camera_id = None
        split = ""
        frame_index = 0
    elif isinstance(value, dict):
        raw_path = str(value.get("path", ""))
        raw_camera = value.get("camera_id")
        camera_id = int(raw_camera) if raw_camera is not None else None
        split = str(value.get("split", ""))
        frame_index = int(value.get("frame_index", 0))
    else:
        raise ValueError("Each manifest image must be a path string or an object")
    if not raw_path:
        raise ValueError("Manifest image path cannot be empty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return ImageRecord(
        path=path.resolve(),
        camera_id=camera_id,
        split=split,
        frame_index=frame_index,
    )


def load_manifest(path: Path, protocol: str) -> list[AnnotationGroup]:
    groups: list[AnnotationGroup] = []
    base_dir = path.resolve().parent
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object")
            raw_images = row.get("images", [])
            if not isinstance(raw_images, list):
                raise ValueError(f"Manifest line {line_number} images must be an array")
            images = [_manifest_image(item, base_dir) for item in raw_images]
            if not images:
                raise ValueError(f"Manifest line {line_number} has no images")
            groups.append(
                AnnotationGroup(
                    dataset=str(row.get("dataset", "custom")),
                    protocol=protocol,
                    pid=int(row["pid"]),
                    clothes_id=(
                        int(row["clothes_id"])
                        if row.get("clothes_id") is not None
                        else None
                    ),
                    modality=str(row.get("modality", "rgb")),
                    group_id=str(row.get("group_id", "")),
                    images=images,
                )
            )
    if not groups:
        raise ValueError("Manifest contains no groups")
    return sorted(groups, key=lambda group: group.resume_key)


def discover_groups(
    *,
    protocol: str,
    dataset: str,
    data_root: Path | None,
    manifest: Path | None,
) -> list[AnnotationGroup]:
    if dataset == "manifest":
        if manifest is None:
            raise ValueError("--manifest is required when --dataset manifest is used")
        return load_manifest(manifest, protocol)
    if data_root is None:
        raise ValueError("--data-root is required for built-in dataset adapters")
    if protocol == "public-rgb":
        return discover_public_rgb(data_root, dataset)
    if protocol == "rgb-ir" and dataset == "atustc":
        return discover_atustc(data_root)
    raise ValueError(f"Dataset {dataset!r} is not available for protocol {protocol!r}")
