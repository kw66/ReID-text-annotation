# Annotation files

All files are UTF-8 JSON Lines. Each line is an independent JSON object. No
source image, filesystem path, service endpoint, API key, or runtime log is
included.

## Public RGB schema

```json
{"pid":1,"dense_en":"..."}
```

| File | Records | SHA-256 |
| --- | ---: | --- |
| `public_rgb/market1501_text_release.jsonl` | 1,501 | `c5671dc3bf5a3bfa32218a69bc66c604fd060de099588f60e3be66e48abc3824` |
| `public_rgb/dukemtmc_text_release.jsonl` | 1,812 | `d550d7300d8af5d41985640809cdf7475f26fe31696b1d08ffede1916634bcac` |
| `public_rgb/cuhk03_text_release.jsonl` | 1,467 | `ba12a7c65c5fbdf6bab2705e11a2cbb528437103c749d8b0df16a18c3a4d746b` |
| `public_rgb/msmt17_text_release.jsonl` | 3,060 | `a5e42de687ca93cd057adcbb97c13ea39e61472b1c26f6ee61bc2b4884e8b763` |

## RGB/IR schema

```json
{
  "pid": 1,
  "clothes_id": 1,
  "modality": "rgb",
  "dense_en": "...",
  "captions": [{"style": "dense_full", "en": "..."}]
}
```

The actual `captions` array has exactly ten objects: five fixed styles and five
diverse styles.

| File | Records | SHA-256 |
| --- | ---: | --- |
| `rgb_ir/atustc_text_release.jsonl` | 1,310 | `36ac285032d0023941bd756cd42358ba9595dc16bf6be348a644156856ed7508` |

Run `python scripts/validate_annotations.py` from the repository root to verify
all schemas, counts, keys, text safety checks, and duplicate constraints.
