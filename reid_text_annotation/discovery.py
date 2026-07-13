from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import AnnotationGroup, ImageRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PUBLIC_RGB_DATASETS = {"market1501", "dukemtmc", "cuhk03", "msmt17"}
TASK_DATASETS = {
    "traditional": PUBLIC_RGB_DATASETS,
    "anytime": {"atustc"},
    "clothes-changing": {"prcc"},
    "visible-infrared": {"sysu-mm01"},
}
_PID_CAMERA_PATTERN = re.compile(r"^(-?\d+)_c(\d+)", re.IGNORECASE)
_ATUSTC_PATTERN = re.compile(
    r"p(\d+)-d(\d+)-c(\d+)[\\/]cam(\d+)-f(\d+)-(\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
_PRCC_FLAT_PATTERN = re.compile(r"(\d+)_c(\d+)_(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_PRCC_TRAIN_PATTERN = re.compile(r"(\d+)/([ABC])_cropped_rgb(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_PRCC_TEST_PATTERN = re.compile(r"([ABC])/(\d+)/cropped_rgb(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_SYSU_RAW_PATTERN = re.compile(r"cam(\d+)/(\d+)/(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_SYSU_FLAT_PATTERN = re.compile(r"(\d+)_c(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


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


def discover_traditional(
    data_root: Path,
    dataset: str,
    annotation_format: str,
) -> list[AnnotationGroup]:
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
            task="traditional",
            annotation_format=annotation_format,
            pid=pid,
            clothes_id=1 if annotation_format == "diverse" else None,
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
        raise ValueError("No valid AT-USTC images were discovered under the dataset root")
    return [
        AnnotationGroup(
            dataset="atustc",
            task="anytime",
            annotation_format="diverse",
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


def _diverse_groups(
    grouped: dict[tuple[int, int, str], list[ImageRecord]],
    *,
    dataset: str,
    task: str,
) -> list[AnnotationGroup]:
    if not grouped:
        raise ValueError(f"No valid images were discovered for {dataset}")
    return [
        AnnotationGroup(
            dataset=dataset,
            task=task,
            annotation_format="diverse",
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


def discover_prcc(data_root: Path) -> list[AnnotationGroup]:
    root = data_root.resolve()
    grouped: dict[tuple[int, int, str], list[ImageRecord]] = defaultdict(list)
    prepared = all((root / name).exists() for name in ("train", "query", "gallery"))
    if prepared:
        for split, directory in (
            ("train", root / "train"),
            ("query", root / "query"),
            ("gallery", root / "gallery"),
        ):
            for path in _image_files(directory):
                match = _PRCC_FLAT_PATTERN.search(path.name)
                if match is None:
                    continue
                pid, camera_id, clothes_id, frame, _suffix = match.groups()
                grouped[(int(pid), int(clothes_id), "rgb")].append(
                    ImageRecord(
                        path=path.resolve(),
                        camera_id=int(camera_id),
                        split=split,
                        frame_index=int(frame),
                    )
                )
        return _diverse_groups(grouped, dataset="prcc", task="clothes-changing")

    camera_clothes = {"A": (1, 1), "B": (2, 1), "C": (3, 2)}
    rgb_root = root / "rgb"
    for directory in (rgb_root / "train", rgb_root / "val"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*/*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(rgb_root).as_posix()
            match = _PRCC_TRAIN_PATTERN.search(relative)
            if match is None:
                continue
            pid, camera_tag, frame, _suffix = match.groups()
            camera_id, clothes_id = camera_clothes[camera_tag.upper()]
            grouped[(int(pid), clothes_id, "rgb")].append(
                ImageRecord(
                    path=path.resolve(),
                    camera_id=camera_id,
                    split="train",
                    frame_index=int(frame),
                )
            )

    test_root = rgb_root / "test"
    if test_root.exists():
        for path in sorted(test_root.glob("*/*/*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(test_root).as_posix()
            match = _PRCC_TEST_PATTERN.search(relative)
            if match is None:
                continue
            camera_tag, pid, frame, _suffix = match.groups()
            camera_id, clothes_id = camera_clothes[camera_tag.upper()]
            split = "gallery" if camera_tag.upper() == "A" else "query"
            grouped[(int(pid), clothes_id, "rgb")].append(
                ImageRecord(
                    path=path.resolve(),
                    camera_id=camera_id,
                    split=split,
                    frame_index=int(frame),
                )
            )
    return _diverse_groups(grouped, dataset="prcc", task="clothes-changing")


def _sysu_available_ids(root: Path) -> set[int] | None:
    path = root / "exp" / "available_id.txt"
    if not path.exists():
        return None
    identifiers: set[int] = set()
    for token in re.split(r"[\s,]+", path.read_text(encoding="utf-8").strip()):
        if token:
            try:
                identifiers.add(int(token))
            except ValueError:
                continue
    return identifiers


def _sysu_modality(camera_id: int) -> str:
    return "rgb" if camera_id in {1, 2, 4, 5} else "ir"


def discover_sysu(data_root: Path) -> list[AnnotationGroup]:
    root = data_root.resolve()
    available_ids = _sysu_available_ids(root)
    grouped: dict[tuple[int, int, str], list[ImageRecord]] = defaultdict(list)
    raw_images = [
        path
        for path in sorted(root.glob("cam*/*/*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if raw_images:
        for path in raw_images:
            relative = path.relative_to(root).as_posix()
            match = _SYSU_RAW_PATTERN.search(relative)
            if match is None:
                continue
            camera_id, pid, frame, _suffix = match.groups()
            pid_value = int(pid)
            if available_ids is not None and pid_value not in available_ids:
                continue
            camera_value = int(camera_id)
            modality = _sysu_modality(camera_value)
            grouped[(pid_value, 1, modality)].append(
                ImageRecord(
                    path=path.resolve(),
                    camera_id=camera_value,
                    split="all",
                    frame_index=int(frame),
                )
            )
        return _diverse_groups(grouped, dataset="sysu-mm01", task="visible-infrared")

    for split, directory in (
        ("train", root / "train"),
        ("query", root / "query"),
        ("gallery", root / "gallery"),
    ):
        for path in _image_files(directory):
            match = _SYSU_FLAT_PATTERN.search(path.name)
            if match is None:
                continue
            pid, camera_id, frame, _suffix = match.groups()
            pid_value = int(pid)
            if available_ids is not None and pid_value not in available_ids:
                continue
            camera_value = int(camera_id)
            modality = _sysu_modality(camera_value)
            grouped[(pid_value, 1, modality)].append(
                ImageRecord(
                    path=path.resolve(),
                    camera_id=camera_value,
                    split=split,
                    frame_index=int(frame),
                )
            )
    return _diverse_groups(grouped, dataset="sysu-mm01", task="visible-infrared")


def discover_groups(
    *,
    task: str,
    dataset: str,
    annotation_format: str,
    data_root: Path,
) -> list[AnnotationGroup]:
    task = task.strip().lower()
    dataset = dataset.strip().lower()
    annotation_format = annotation_format.strip().lower()
    if task not in TASK_DATASETS or dataset not in TASK_DATASETS[task]:
        raise ValueError(f"Dataset {dataset!r} does not belong to task {task!r}")
    if task == "traditional":
        if annotation_format == "diverse" and dataset != "market1501":
            raise ValueError("The released diverse traditional format is available only for Market-1501")
        return discover_traditional(data_root, dataset, annotation_format)
    if annotation_format != "diverse":
        raise ValueError(f"Task {task!r} requires --format diverse")
    if task == "anytime":
        return discover_atustc(data_root)
    if task == "clothes-changing":
        return discover_prcc(data_root)
    return discover_sysu(data_root)
