from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .api import CompatibleAPI
from .models import AnnotationGroup
from .prompts import BRANCH_INSTRUCTIONS, SYSTEM_PROMPT, evidence_prompt, synthesis_prompt
from .selection import image_data_url, select_images


def _clean_string_list(value: object, limit: int = 24) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _clean_dense_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Model output is missing dense_en")
    text = " ".join(value.split()).strip()
    if not text:
        raise ValueError("Model output contains an empty dense_en")
    return text


_CAPTION_STYLES = [
    "dense_full",
    "person_core",
    "outfit_core",
    "distinctive_mix",
    "retrieval_balanced",
    "random_a",
    "random_b",
    "random_c",
    "random_d",
    "random_e",
]


def _clean_caption_objects(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(_CAPTION_STYLES):
        raise ValueError("Diverse model output must contain exactly 10 captions")
    by_style: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each diverse caption must be an object")
        style = str(item.get("style", "")).strip()
        text = " ".join(str(item.get("en", "")).split()).strip()
        if style not in _CAPTION_STYLES or not text or style in by_style:
            raise ValueError("Diverse caption objects contain invalid style or text")
        by_style[style] = text
    if set(by_style) != set(_CAPTION_STYLES):
        raise ValueError("Diverse caption styles are incomplete")
    return [{"style": style, "en": by_style[style]} for style in _CAPTION_STYLES]


@dataclass(frozen=True)
class AnnotationPipeline:
    api: CompatibleAPI
    max_images: int = 12
    image_max_side: int = 1024

    def annotate_group(self, group: AnnotationGroup) -> dict[str, Any]:
        selected = select_images(group.images, self.max_images)
        image_urls = [image_data_url(item, self.image_max_side) for item in selected]

        evidence: dict[str, Any] = {}
        for branch in BRANCH_INSTRUCTIONS:
            response = self.api.complete_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=evidence_prompt(branch, group.modality, group.task),
                image_urls=image_urls,
                temperature=0.0,
            )
            evidence[branch] = {
                "cues": _clean_string_list(response.get("cues")),
                "uncertain_cues": _clean_string_list(response.get("uncertain_cues")),
            }

        final = self.api.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=synthesis_prompt(
                annotation_format=group.annotation_format,
                modality=group.modality,
                task=group.task,
                evidence=evidence,
            ),
            temperature=0.25 if group.annotation_format == "diverse" else 0.1,
        )
        dense_en = _clean_dense_text(final.get("dense_en"))
        if group.annotation_format == "dense":
            return {"pid": group.pid, "dense_en": dense_en}

        captions = _clean_caption_objects(final.get("captions"))
        return {
            "pid": group.pid,
            "clothes_id": int(group.clothes_id),
            "modality": group.modality,
            "dense_en": dense_en,
            "captions": captions,
        }


def _record_key(record: dict[str, Any], annotation_format: str) -> tuple[int | str, ...]:
    if annotation_format == "dense":
        return (int(record["pid"]),)
    return (
        int(record["pid"]),
        int(record["clothes_id"]),
        str(record["modality"]).lower(),
    )


def _read_existing(
    path: Path,
    annotation_format: str,
) -> dict[tuple[int | str, ...], dict[str, Any]]:
    records: dict[tuple[int | str, ...], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Existing output contains a non-object JSONL row")
            records[_record_key(row, annotation_format)] = row
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _normalize_output(path: Path, annotation_format: str) -> None:
    records = _read_existing(path, annotation_format)
    ordered = [records[key] for key in sorted(records)]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _safe_error(group: AnnotationGroup, exc: BaseException) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "pid": group.pid,
        "clothes_id": group.clothes_id,
        "modality": group.modality,
        "error_type": type(exc).__name__,
    }


def dry_run(groups: Iterable[AnnotationGroup], max_images: int) -> dict[str, int]:
    group_count = 0
    image_count = 0
    selected_count = 0
    unreadable_groups = 0
    for group in groups:
        group_count += 1
        image_count += len(group.images)
        try:
            selected_count += len(select_images(group.images, max_images))
        except ValueError:
            unreadable_groups += 1
    return {
        "groups": group_count,
        "images": image_count,
        "selected_images": selected_count,
        "unreadable_groups": unreadable_groups,
    }


def run_annotation(
    *,
    groups: list[AnnotationGroup],
    pipeline: AnnotationPipeline,
    output_path: Path,
    annotation_format: str,
    workers: int = 1,
) -> dict[str, int]:
    existing = _read_existing(output_path, annotation_format)
    pending = [group for group in groups if group.resume_key not in existing]
    errors_path = output_path.with_suffix(".errors.jsonl")
    completed = 0
    failed = 0

    def save_result(group: AnnotationGroup, result: dict[str, Any] | None, exc: BaseException | None) -> None:
        nonlocal completed, failed
        if exc is not None:
            _append_jsonl(errors_path, _safe_error(group, exc))
            failed += 1
            return
        if result is None:
            raise RuntimeError("Annotation worker returned no result")
        _append_jsonl(output_path, result)
        completed += 1

    if workers <= 1:
        for group in pending:
            try:
                save_result(group, pipeline.annotate_group(group), None)
            except Exception as exc:  # Keep long annotation runs resumable.
                save_result(group, None, exc)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(pipeline.annotate_group, group): group
                for group in pending
            }
            for future in as_completed(futures):
                group = futures[future]
                try:
                    save_result(group, future.result(), None)
                except Exception as exc:  # Keep long annotation runs resumable.
                    save_result(group, None, exc)

    if output_path.exists():
        _normalize_output(output_path, annotation_format)
    return {
        "groups": len(groups),
        "already_complete": len(existing),
        "completed": completed,
        "failed": failed,
    }
