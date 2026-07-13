from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api import CompatibleAPI
from .discovery import discover_groups
from .env import load_env_file
from .pipeline import AnnotationPipeline, dry_run, run_annotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate text annotations for traditional, anytime, changing-clothes, "
            "and visible-infrared person ReID datasets."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["traditional", "anytime", "clothes-changing", "visible-infrared"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["market1501", "dukemtmc", "cuhk03", "msmt17", "atustc", "prcc", "sysu-mm01"],
    )
    parser.add_argument("--format", dest="annotation_format", choices=["dense", "diverse"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument(
        "--json-response-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request JSON mode; disable it for providers that do not support response_format.",
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    annotation_format = args.annotation_format
    if annotation_format is None:
        annotation_format = "dense" if args.task == "traditional" else "diverse"
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.image_max_side <= 0:
        parser.error("--image-max-side must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")

    try:
        groups = discover_groups(
            task=args.task,
            dataset=args.dataset,
            annotation_format=annotation_format,
            data_root=args.data_root,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    if args.dry_run:
        summary = dry_run(groups, args.max_images)
        print(json.dumps({"status": "dry_run", **summary}, ensure_ascii=False))
        return

    base_url = (args.api_base_url or os.getenv("REID_API_BASE_URL", "")).strip()
    api_key = os.getenv("REID_API_KEY", "").strip()
    model = (args.model or os.getenv("REID_API_MODEL", "")).strip()
    if not base_url or not api_key or not model:
        raise SystemExit(
            "Configure REID_API_BASE_URL, REID_API_KEY, and REID_API_MODEL in "
            "the environment or a local .env file."
        )

    api = CompatibleAPI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        max_output_tokens=args.max_output_tokens,
        use_json_response_format=args.json_response_format,
    )
    pipeline = AnnotationPipeline(
        api=api,
        max_images=args.max_images,
        image_max_side=args.image_max_side,
    )
    summary = run_annotation(
        groups=groups,
        pipeline=pipeline,
        output_path=args.output,
        annotation_format=annotation_format,
        workers=max(1, args.workers),
    )
    print(json.dumps({"status": "complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
