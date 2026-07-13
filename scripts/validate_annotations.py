from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPECS = {
    "public_rgb/market1501_text_release.jsonl": (1501, {"pid", "dense_en"}, "public-rgb"),
    "public_rgb/dukemtmc_text_release.jsonl": (1812, {"pid", "dense_en"}, "public-rgb"),
    "public_rgb/cuhk03_text_release.jsonl": (1467, {"pid", "dense_en"}, "public-rgb"),
    "public_rgb/msmt17_text_release.jsonl": (3060, {"pid", "dense_en"}, "public-rgb"),
    "rgb_ir/atustc_text_release.jsonl": (
        1310,
        {"pid", "clothes_id", "modality", "dense_en", "captions"},
        "rgb-ir",
    ),
}

FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|[A-Za-z]:[\\/]|/(?:home|Users)/|api[_-]?key|authorization\s*:|bearer\s+|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)

CAPTION_STYLES = [
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


def _key(row: dict[str, Any], protocol: str) -> tuple[int | str, ...]:
    if protocol == "public-rgb":
        return (int(row["pid"]),)
    return (int(row["pid"]), int(row["clothes_id"]), str(row["modality"]))


def validate_file(path: Path, expected_count: int, keys: set[str], protocol: str) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[int | str, ...]] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            count += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                errors.append(f"{path.name}:{line_number}: invalid JSON")
                continue
            if not isinstance(row, dict) or set(row) != keys:
                errors.append(f"{path.name}:{line_number}: unexpected schema")
                continue
            dense_en = row.get("dense_en")
            if not isinstance(dense_en, str) or not dense_en.strip():
                errors.append(f"{path.name}:{line_number}: empty dense_en")
            if protocol == "rgb-ir":
                captions = row.get("captions")
                if not isinstance(captions, list) or len(captions) != 10:
                    errors.append(f"{path.name}:{line_number}: captions must contain 10 objects")
                elif any(
                    not isinstance(item, dict)
                    or set(item) != {"style", "en"}
                    or not isinstance(item.get("style"), str)
                    or not isinstance(item.get("en"), str)
                    or not item["en"].strip()
                    for item in captions
                ):
                    errors.append(f"{path.name}:{line_number}: invalid caption")
                elif [item["style"] for item in captions] != CAPTION_STYLES:
                    errors.append(f"{path.name}:{line_number}: invalid caption style order")
                if row.get("modality") not in {"rgb", "ir"}:
                    errors.append(f"{path.name}:{line_number}: invalid modality")
            text_values = [value for value in row.values() if isinstance(value, str)]
            text_values.extend(
                item.get("en", "")
                for item in row.get("captions", [])
                if isinstance(item, dict)
            )
            if any(FORBIDDEN_TEXT.search(value) for value in text_values):
                errors.append(f"{path.name}:{line_number}: forbidden private or endpoint text")
            try:
                record_key = _key(row, protocol)
            except (KeyError, TypeError, ValueError):
                errors.append(f"{path.name}:{line_number}: invalid key fields")
                continue
            if record_key in seen:
                errors.append(f"{path.name}:{line_number}: duplicate record key")
            seen.add(record_key)
    if count != expected_count:
        errors.append(f"{path.name}: expected {expected_count} rows, found {count}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate released annotation JSONL files.")
    parser.add_argument("--root", type=Path, default=Path("annotations"))
    args = parser.parse_args()

    all_errors: list[str] = []
    for relative, (count, keys, protocol) in SPECS.items():
        path = args.root / relative
        if not path.exists():
            all_errors.append(f"missing: {relative}")
            continue
        all_errors.extend(validate_file(path, count, keys, protocol))
    if all_errors:
        for error in all_errors:
            print(error)
        raise SystemExit(1)
    total = sum(spec[0] for spec in SPECS.values())
    print(f"Validated {len(SPECS)} files and {total} annotation records.")


if __name__ == "__main__":
    main()
