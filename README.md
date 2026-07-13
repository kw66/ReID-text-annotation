<h1 align="center">ReID Text Annotation</h1>

<p align="center">
  Text annotations and generation code for four person Re-identification tasks.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/annotations-12%2C075-2E8B57" alt="12,075 annotations">
</p>

## 🧭 Task Map

<table width="100%">
  <tr>
    <th align="center">🏙️ Traditional ReID</th>
    <th align="center">🕒 Anytime ReID</th>
    <th align="center">👕 Changing-Clothes ReID</th>
    <th align="center">🌗 Visible-Infrared ReID</th>
  </tr>
  <tr>
    <td align="center">Market-1501<br>DukeMTMC-reID<br>CUHK03-NP<br>MSMT17</td>
    <td align="center">AT-USTC</td>
    <td align="center">PRCC</td>
    <td align="center">SYSU-MM01</td>
  </tr>
  <tr>
    <td align="center">RGB, same-clothes benchmark</td>
    <td align="center">Day/night, short/long term</td>
    <td align="center">Identity retrieval across outfits</td>
    <td align="center">RGB ↔ infrared retrieval</td>
  </tr>
</table>

The task categories are kept explicit throughout the CLI and release folders:

| Task | What changes between query and gallery | Built-in dataset |
| --- | --- | --- |
| **Traditional ReID** | Camera and viewpoint; usually the same clothes and RGB modality | Market-1501, DukeMTMC-reID, CUHK03-NP, MSMT17 |
| **Anytime ReID** | Day/night, short/long time span, clothes, scene, and modality | AT-USTC |
| **Changing-Clothes ReID** | Clothing appearance | PRCC |
| **Visible-Infrared ReID** | Visible and infrared modalities | SYSU-MM01 |

## 📦 Released Annotations

| Task | Dataset / version | Annotation unit | Format | Records | File |
| --- | --- | --- | --- | ---: | --- |
| Traditional | Market-1501 **PID dense** | `pid` | 1 dense text | 1,501 | [`market1501_pid_dense.jsonl`](annotations/traditional/market1501_pid_dense.jsonl) |
| Traditional | Market-1501 **AT-USTC-style** | `pid, clothes_id, modality` | 1 dense + 10 captions | 1,501 | [`market1501_atustc_style.jsonl`](annotations/traditional/market1501_atustc_style.jsonl) |
| Traditional | DukeMTMC-reID | `pid` | 1 dense text | 1,812 | [`dukemtmc_pid_dense.jsonl`](annotations/traditional/dukemtmc_pid_dense.jsonl) |
| Traditional | CUHK03-NP | `pid` | 1 dense text | 1,467 | [`cuhk03_pid_dense.jsonl`](annotations/traditional/cuhk03_pid_dense.jsonl) |
| Traditional | MSMT17 | `pid` | 1 dense text | 3,060 | [`msmt17_pid_dense.jsonl`](annotations/traditional/msmt17_pid_dense.jsonl) |
| Anytime | AT-USTC | `pid, clothes_id, modality` | 1 dense + 10 captions | 1,310 | [`atustc_group_diverse.jsonl`](annotations/anytime/atustc_group_diverse.jsonl) |
| Changing-clothes | PRCC | `pid, clothes_id, modality` | 1 dense + 10 captions | 442 | [`prcc_group_diverse.jsonl`](annotations/clothes_changing/prcc_group_diverse.jsonl) |
| Visible-infrared | SYSU-MM01 | `pid, clothes_id, modality` | 1 dense + 10 captions | 982 | [`sysu_mm01_group_diverse.jsonl`](annotations/visible_infrared/sysu_mm01_group_diverse.jsonl) |

### Market-1501 has two versions

| Version | Use it when | Output schema |
| --- | --- | --- |
| **PID dense** | Training or evaluating Market-1501 as a conventional RGB ReID dataset | `pid, dense_en` |
| **AT-USTC-style** | Combining Market-1501 with AT-USTC, PRCC, or SYSU-MM01 under one group-level text interface | `pid, clothes_id, modality, dense_en, captions` |

The two files contain the same 1,501 Market identities but serve different
training interfaces. They should not be concatenated as independent identities.

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/kw66/ReID-text-annotation.git
cd ReID-text-annotation
python -m pip install -e .
```

### 2. Configure your own multimodal API

Create a local `.env` and provide your own values:

```dotenv
REID_API_BASE_URL=<your-compatible-api-base-url>
REID_API_KEY=<your-api-key>
REID_API_MODEL=<your-multimodal-model>
```

The API must provide an OpenAI-compatible Chat Completions endpoint with image
input support. `.env` is ignored by Git, and the API key has no command-line
option.

## ▶️ Run Commands

### 🏙️ Traditional ReID

Market-1501, PID-level dense text:

```bash
python -m reid_text_annotation \
  --task traditional \
  --dataset market1501 \
  --format dense \
  --data-root /path/to/Market-1501 \
  --output outputs/market1501_pid_dense.jsonl
```

Market-1501, AT-USTC-compatible diverse text:

```bash
python -m reid_text_annotation \
  --task traditional \
  --dataset market1501 \
  --format diverse \
  --data-root /path/to/Market-1501 \
  --output outputs/market1501_atustc_style.jsonl
```

For the other traditional datasets, replace `--dataset` and `--data-root`:

```bash
python -m reid_text_annotation --task traditional --dataset dukemtmc --format dense --data-root /path/to/DukeMTMC-reID --output outputs/dukemtmc_pid_dense.jsonl
python -m reid_text_annotation --task traditional --dataset cuhk03 --format dense --data-root /path/to/CUHK03-NP --output outputs/cuhk03_pid_dense.jsonl
python -m reid_text_annotation --task traditional --dataset msmt17 --format dense --data-root /path/to/MSMT17 --output outputs/msmt17_pid_dense.jsonl
```

### 🕒 Anytime ReID

```bash
python -m reid_text_annotation \
  --task anytime \
  --dataset atustc \
  --format diverse \
  --data-root /path/to/AT-USTC \
  --output outputs/atustc_group_diverse.jsonl
```

### 👕 Changing-Clothes ReID

```bash
python -m reid_text_annotation \
  --task clothes-changing \
  --dataset prcc \
  --format diverse \
  --data-root /path/to/PRCC \
  --output outputs/prcc_group_diverse.jsonl
```

### 🌗 Visible-Infrared ReID

```bash
python -m reid_text_annotation \
  --task visible-infrared \
  --dataset sysu-mm01 \
  --format diverse \
  --data-root /path/to/SYSU-MM01 \
  --output outputs/sysu_mm01_group_diverse.jsonl
```

Append `--dry-run --max-groups 10` to any command to check dataset discovery and
image readability without calling the API.

## 🗂️ Expected Dataset Layouts

| Dataset | Accepted layout |
| --- | --- |
| Market-1501 / DukeMTMC / CUHK03 / MSMT17 | `bounding_box_train/`, `query/`, `bounding_box_test/` |
| AT-USTC | `p{pid}-d{day}-c{clothes}/cam{camera}-f{slot}-{frame}.jpg` |
| PRCC | Original `rgb/train`, `rgb/val`, `rgb/test` layout, or prepared `train`, `query`, `gallery` folders |
| SYSU-MM01 | Original `cam1`-`cam6` layout, or prepared `train`, `query`, `gallery` folders |

AT-USTC cameras 1-8 are treated as RGB and cameras 9-16 as infrared. SYSU-MM01
cameras 1, 2, 4, and 5 are RGB; cameras 3 and 6 are infrared.

## 🧾 Output Schemas

### Dense format

```json
{"pid":1,"dense_en":"..."}
```

### Diverse format

```json
{
  "pid": 1,
  "clothes_id": 1,
  "modality": "rgb",
  "dense_en": "...",
  "captions": [
    {"style": "dense_full", "en": "..."},
    {"style": "person_core", "en": "..."},
    {"style": "outfit_core", "en": "..."},
    {"style": "distinctive_mix", "en": "..."},
    {"style": "retrieval_balanced", "en": "..."},
    {"style": "random_a", "en": "..."},
    {"style": "random_b", "en": "..."},
    {"style": "random_c", "en": "..."},
    {"style": "random_d", "en": "..."},
    {"style": "random_e", "en": "..."}
  ]
}
```

## ⚙️ Annotation Pipeline

1. Group images using the task-specific identity, clothing, and modality keys.
2. Select readable views with camera, split, and visual-diversity coverage.
3. Extract overview, head/face, body, and garment/item evidence.
4. Remove uncertain cues and generate dense or ten-style diverse text.

IR groups never use RGB color evidence. Changing-clothes groups never merge
garments from different clothing labels.

## 📝 Citation

If this repository helps your research, please cite:

```bibtex
@inproceedings{wang2026pid,
  title     = {PID-Level Dense Text and Attribute-Indexed Semantic Prototypes for Exemplar-Free Lifelong Person Re-Identification},
  author    = {Wang, Chenyang and Liu, Bin and Li, Xulin and Yu, Nenghai},
  booktitle = {Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  year      = {2026}
}
```
