# ReID Text Annotation

Open annotation files and a reproducible, privacy-conscious pipeline for two
person re-identification text protocols:

- **Public RGB ReID:** one dense English description per person identity (PID).
- **Changing-clothes RGB/IR ReID:** one dense description plus ten diverse
  captions per `(pid, clothes_id, modality)` group.

The repository contains text annotations and annotation code only. It does not
contain source images, private dataset roots, runtime logs, service endpoints,
API keys, or machine-specific configuration.

## Released annotations

| Protocol | Dataset | Records | File |
| --- | --- | ---: | --- |
| PID-level public RGB | Market-1501 | 1,501 | `annotations/public_rgb/market1501_text_release.jsonl` |
| PID-level public RGB | DukeMTMC-reID | 1,812 | `annotations/public_rgb/dukemtmc_text_release.jsonl` |
| PID-level public RGB | CUHK03-NP | 1,467 | `annotations/public_rgb/cuhk03_text_release.jsonl` |
| PID-level public RGB | MSMT17 | 3,060 | `annotations/public_rgb/msmt17_text_release.jsonl` |
| Group-level RGB/IR | AT-USTC | 1,310 | `annotations/rgb_ir/atustc_text_release.jsonl` |

There are 9,150 released records in total. Checksums and exact schemas are
documented in [`annotations/README.md`](annotations/README.md).

## Method

Both protocols use the same evidence-first design:

1. Group images by the annotation unit.
2. Select readable and complementary views using image quality, camera/split
   coverage, and low-resolution appearance diversity.
3. Extract structured evidence through four multimodal-model branches:
   overview, head/face, body, and garment-structure/items.
4. Remove uncertain or conflicting cues and synthesize release text.

RGB and infrared groups are processed separately. IR prompts explicitly forbid
color claims. See [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md) for the full protocol.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/kw66/ReID-text-annotation.git
cd ReID-text-annotation
python -m pip install -e .
```

The only runtime dependency is Pillow. API requests use the Python standard
library.

## API configuration

The pipeline works with an OpenAI-compatible Chat Completions service that
accepts image data URLs. Create a local `.env` from `.env.example` and fill in
your own values:

```dotenv
REID_API_BASE_URL=<your-compatible-api-base-url>
REID_API_KEY=<your-api-key>
REID_API_MODEL=<your-multimodal-model>
```

`.env` is ignored by Git. The API key is read only from the environment or the
local `.env` file; there is no command-line key option, so it is not exposed in
process arguments. A base URL may be either the full `/chat/completions`
endpoint or a base path to which `/chat/completions` should be appended.

## Public RGB annotation

Built-in adapters support Market-1501, DukeMTMC-reID, CUHK03-NP, and MSMT17.
They expect the common `bounding_box_train`, `query`, and `bounding_box_test`
directories.

```bash
python -m reid_text_annotation \
  --protocol public-rgb \
  --dataset market1501 \
  --data-root /path/to/dataset \
  --output outputs/market1501_text.jsonl
```

Use `--dataset dukemtmc`, `--dataset cuhk03`, or `--dataset msmt17` for the
other built-in adapters. A no-API validation pass is available:

```bash
python -m reid_text_annotation \
  --protocol public-rgb \
  --dataset market1501 \
  --data-root /path/to/dataset \
  --output outputs/dry_run.jsonl \
  --max-groups 10 \
  --dry-run
```

## RGB/IR changing-clothes annotation

The built-in AT-USTC adapter recognizes this relative naming convention:

```text
p{pid}-d{day}-c{clothes_id}/cam{camera_id}-f{slot}-{frame}.jpg
```

Cameras 1-8 are treated as RGB and cameras 9 or higher as IR, matching the
dataset protocol used for the released annotation file.

```bash
python -m reid_text_annotation \
  --protocol rgb-ir \
  --dataset atustc \
  --data-root /path/to/dataset \
  --output outputs/atustc_text.jsonl
```

For any other layout, use a manifest. Paths in the manifest are resolved
relative to the manifest file:

```bash
python -m reid_text_annotation \
  --protocol rgb-ir \
  --dataset manifest \
  --manifest examples/rgb_ir_manifest.example.jsonl \
  --output outputs/custom_rgb_ir_text.jsonl
```

The public and RGB/IR manifest schemas are shown in `examples/`.

## Output schemas

Public RGB records contain no image paths or runtime metadata:

```json
{"pid":1,"dense_en":"..."}
```

RGB/IR records preserve the group key and ten named caption styles:

```json
{
  "pid": 1,
  "clothes_id": 1,
  "modality": "rgb",
  "dense_en": "...",
  "captions": [
    {"style": "dense_full", "en": "..."},
    {"style": "person_core", "en": "..."}
  ]
}
```

The full list contains five fixed styles (`dense_full`, `person_core`,
`outfit_core`, `distinctive_mix`, `retrieval_balanced`) and five diverse styles
(`random_a` through `random_e`).

Runs are resumable: existing record keys are skipped, successful rows are
written incrementally, and failures are recorded without exception messages or
local paths. Use `--workers` cautiously because higher concurrency can increase
API cost and rate-limit pressure.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_annotations.py
```

The release validator checks record counts, schemas, duplicate group keys,
caption structure, modalities, and accidental endpoint/path/credential text.

## Privacy and dataset terms

- No source image is redistributed by this repository.
- Images selected during annotation are sent to the API provider configured by
  the user. Review that provider's retention and privacy terms before use.
- Do not use the pipeline to infer names, ethnicity, nationality, occupation,
  health, religion, or other unsupported sensitive attributes.
- ReID datasets can have access restrictions and privacy requirements. Obtain
  each dataset from its authorized source and follow its current terms.
- DukeMTMC and its derivatives have had access restrictions; this repository
  provides no images or download mirror.

The public RGB datasets correspond to the established Market-1501, CUHK03,
DukeMTMC-reID, and MSMT17 benchmarks. Their foundational references are:

- Zheng et al., *Scalable Person Re-identification: A Benchmark*, ICCV 2015.
- Li et al., *DeepReID: Deep Filter Pairing Neural Network for Person
  Re-identification*, CVPR 2014.
- Ristani et al., *Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking*, ECCV Workshops 2016.
- Wei et al., *Person Transfer GAN to Bridge Domain Gap for Person
  Re-Identification*, CVPR 2018.

## License

Code is released under the MIT License. Annotation JSONL files are released
under CC BY 4.0, subject to the separate terms of the source image datasets.
See [`DATA_LICENSE.md`](DATA_LICENSE.md).
