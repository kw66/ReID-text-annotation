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
        description="Generate group-level text annotations for person ReID datasets.",
        allow_abbrev=False,
    )
    parser.add_argument("--protocol", required=True, choices=["public-rgb", "rgb-ir"])
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["market1501", "dukemtmc", "cuhk03", "msmt17", "atustc", "manifest"],
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
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
    args = build_parser().parse_args()
    load_env_file(args.env_file)

    groups = discover_groups(
        protocol=args.protocol,
        dataset=args.dataset,
        data_root=args.data_root,
        manifest=args.manifest,
    )
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
        protocol=args.protocol,
        workers=max(1, args.workers),
    )
    print(json.dumps({"status": "complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
