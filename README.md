# Image Dataset Creator (`idc`)

![CI](https://github.com/yourname/image-dataset-creator/actions/workflows/test.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Build high-quality, legally-safe image datasets from commercial-friendly sources — in one command.

```bash
idc search "golden retriever" --count 500 --format yolo --output ./dogs
```

---

## Features

- Search across **Unsplash, Pexels, Pixabay, Wikimedia Commons, and Open Images**
- Only downloads images with **commercial-safe licenses** (CC0, Unsplash, Pexels, Pixabay, CC-BY)
- **Attribution tracking** — every image has photographer, license, and source URL stored
- **Quality filtering** — minimum resolution, aspect ratio, file size, and blur detection
- **Perceptual deduplication** — pHash removes near-duplicate images
- **CLIP semantic filtering** — score images against the query using a CLIP model (optional)
- **Export formats** — raw folder, YOLO, COCO, HuggingFace datasets, CSV
- **Train/val/test splits** — configurable fractions across all export formats
- **Async downloads** — `asyncio`-based concurrent downloads with configurable concurrency
- **Rate limiting** — per-source configurable delay between paginated API requests
- **Resume downloads** — skips already-downloaded images automatically
- **Streaming export** — memory-efficient record iteration via `Manifest.iter_all()`
- **Download summary** — Rich table showing accepted, skipped, and failed counts after every run
- **Dataset merging** — combine two datasets with cross-dataset deduplication
- **Python API** — composable pipeline for scripting and CI/CD
- **Typed library** — ships with `py.typed` for full type-checker support (PEP 561)

---

## Installation

```bash
pip install image-dataset-creator
```

Optional extras:

```bash
pip install "image-dataset-creator[huggingface]"  # HuggingFace export
pip install "image-dataset-creator[clip]"          # CLIP semantic filtering
```

---

## Quick Start

### 1. Set API keys

Get free API keys from each source:
- **Unsplash**: https://unsplash.com/developers
- **Pexels**: https://www.pexels.com/api/
- **Pixabay**: https://pixabay.com/api/docs/

```bash
idc config set --unsplash-key YOUR_KEY --pexels-key YOUR_KEY --pixabay-key YOUR_KEY
```

Or use environment variables:

```bash
export IDC_UNSPLASH_KEY=your_key
export IDC_PEXELS_KEY=your_key
export IDC_PIXABAY_KEY=your_key
```

Wikimedia Commons and Open Images require no API key.

### 2. Download a dataset

```bash
idc search "golden retriever" --count 500 --output ./dogs
```

### 3. Export to your format

```bash
idc export ./dogs --format yolo --output ./dogs_yolo
```

---

## CLI Reference

### `idc search`

Search and download images in one step.

```bash
idc search QUERY [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--sources` | `unsplash,pexels` | Comma-separated sources: `unsplash`, `pexels`, `pixabay`, `wikimedia`, `openimages` |
| `--count` | `100` | Total images to download |
| `--min-width` | `256` | Minimum image width (px) |
| `--min-height` | `256` | Minimum image height (px) |
| `--max-blur` | `100.0` | Min sharpness (Laplacian variance; images below rejected) |
| `--max-aspect-ratio` | `4.0` | Maximum width/height ratio |
| `--format` | `raw` | Export format: `raw`, `yolo`, `coco`, `huggingface`, `csv` |
| `--val-split` | `0.1` | Fraction reserved for validation split |
| `--test-split` | `0.0` | Fraction reserved for test split |
| `--output` | `./dataset` | Output directory |
| `--label-by-query` | off | Put each comma-separated query in its own subfolder |
| `--dedup` / `--no-dedup` | on | Perceptual deduplication |
| `--clip-filter` | off | Filter by CLIP semantic similarity (requires `[clip]` extra) |
| `--clip-threshold` | `0.2` | Cosine similarity threshold for CLIP filter |
| `--no-resume` | off | Re-download even if already in manifest |
| `--failure-log` | off | Save `download_failures.jsonl` on completion |
| `--workers` | `8` | Max concurrent async downloads |

**Examples:**

```bash
# Download 200 images from all three API sources
idc search "tabby cat" --sources unsplash,pexels,pixabay --count 200

# Multi-class dataset — each query gets its own folder
idc search "golden retriever,labrador,poodle" \
  --count 300 \
  --label-by-query \
  --output ./dog_breeds

# High-quality YOLO export with train/val/test split
idc search "street photography" \
  --min-width 640 \
  --min-height 640 \
  --max-blur 150 \
  --format yolo \
  --val-split 0.1 \
  --test-split 0.1 \
  --output ./street_dataset

# Use Open Images (no API key required)
idc search "dog" --sources openimages --count 100

# Include Wikimedia Commons (CC0 and CC-BY only)
idc search "architecture" --sources unsplash,wikimedia --count 100

# CLIP-based semantic filtering (rejects off-topic images)
idc search "golden retriever" --clip-filter --clip-threshold 0.25

# Log download failures for debugging
idc search "cats" --count 200 --failure-log
```

### `idc config`

```bash
# Set API keys
idc config set --unsplash-key KEY --pexels-key KEY --pixabay-key KEY

# Show which keys are configured
idc config show
```

### `idc filter`

Apply quality filters to an already-downloaded dataset.

```bash
idc filter ./dataset [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--min-width` | `256` | Minimum width |
| `--min-height` | `256` | Minimum height |
| `--max-blur` | `100.0` | Sharpness threshold |
| `--dedup` / `--no-dedup` | off | Remove near-duplicates |
| `--keep-files` | off | Don't delete rejected image files |
| `--dry-run` | off | Preview removals without changing anything |

```bash
# Preview what would be removed
idc filter ./dataset --min-width 512 --dry-run

# Remove blurry and duplicate images
idc filter ./dataset --max-blur 150 --dedup

# Remove from manifest but keep files on disk
idc filter ./dataset --min-width 512 --keep-files
```

### `idc merge`

Combine two datasets into one, with optional cross-dataset deduplication.

```bash
idc merge SOURCE_A SOURCE_B --output MERGED_DIR [--dedup/--no-dedup]
```

```bash
# Merge two datasets and deduplicate
idc merge ./dogs_unsplash ./dogs_pexels --output ./dogs_merged

# Merge without deduplication
idc merge ./dataset_v1 ./dataset_v2 --no-dedup --output ./dataset_combined
```

Images are copied to `MERGED_DIR/images/`. The merged manifest is written to `MERGED_DIR/manifest.db`.

### `idc export`

Re-export an existing dataset to a different format.

```bash
idc export ./dataset --format huggingface --output ./hf_dataset
idc export ./dataset --format coco --val-split 0.2 --output ./coco_dataset
idc export ./dataset --format yolo --val-split 0.1 --test-split 0.1 --output ./yolo_dataset
idc export ./dataset --format csv --output ./csv_export
```

### `idc report`

Generate a CSV attribution report (useful for legal teams).

```bash
idc report ./dataset --output attribution.csv
```

Output columns: `id`, `attribution`, `license_type`, `license_url`, `photographer`, `photographer_url`, `url`, `source`, `query`, `local_path`

### `idc info`

Show dataset statistics.

```bash
idc info ./dataset
```

```
Dataset Info: ./dataset
┌──────────────┬──────────────────────────────┐
│ Metric       │ Value                        │
├──────────────┼──────────────────────────────┤
│ Total images │ 487                          │
│ Sources      │ unsplash: 200, pexels: 187,  │
│              │ pixabay: 100                 │
│ Licenses     │ unsplash: 200, pexels: 187,  │
│              │ pixabay: 100                 │
│ Avg width    │ 1142px                       │
│ Avg height   │ 856px                        │
└──────────────┴──────────────────────────────┘
```

---

## Download Summary

After every `download()` call, `idc` prints a summary table:

```
  Download Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Outcome                           ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Accepted                          │   412 │
│ Skipped (already downloaded)      │    50 │
│ Failed (download error)           │    18 │
│ Rejected (quality)                │    12 │
│ Rejected (duplicate)              │     8 │
└───────────────────────────────────┴───────┘
```

Use `--failure-log` to write `download_failures.jsonl` with per-image failure reasons.

---

## Export Formats

### `raw` (default)

```
dataset/
├── images/
│   ├── unsplash_abc123.jpg
│   └── pexels_456789.jpg
├── metadata.jsonl          # one JSON record per line
└── manifest.db
```

With splits (`--val-split 0.2 --test-split 0.1`):

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata.jsonl          # includes "split" field per record
```

### `yolo`

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/               # if --test-split > 0
├── labels/
│   ├── train/              # empty .txt files (ready for annotation)
│   ├── val/
│   └── test/
└── dataset.yaml
```

### `coco`

```
dataset/
├── images/
└── annotations/
    └── instances.json      # COCO-format with attribution metadata
```

With splits:

```
dataset/
├── images/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json           # if --test-split > 0
```

### `huggingface`

```
dataset/
├── dataset_info.json
└── data-*.arrow            # load with datasets.load_from_disk()
```

```python
from datasets import load_from_disk
ds = load_from_disk("./dataset")
```

### `csv`

```
dataset/
└── dataset.csv             # flat CSV with all metadata columns
```

---

## Python API

### Basic pipeline

```python
from idc import DatasetBuilder, UnsplashSource, PexelsSource
from idc.filters import QualityFilter, Deduplicator
from idc.exporters import YOLOExporter

builder = DatasetBuilder(output_dir="./dogs", dataset_name="dog_breeds")

builder.add_source(UnsplashSource(api_key="your_unsplash_key"))
builder.add_source(PexelsSource(api_key="your_pexels_key"))

builder.add_filter(QualityFilter(
    min_width=640,
    min_height=640,
    blur_threshold=100.0,
    max_aspect_ratio=4.0,
    min_file_size=10_000,
))
builder.add_filter(Deduplicator(threshold=10))

records = builder.search("golden retriever", count=500)
accepted = builder.download(records)
print(f"Accepted {len(accepted)} / {len(records)} images")

# Inspect the summary
summary = builder.last_summary
print(f"Failed downloads: {summary.failed_download}")
print(f"Rejected (quality): {summary.failed_quality}")

builder.export(YOLOExporter(class_names=["dog"], val_split=0.1, test_split=0.1))
```

### Multi-class dataset

```python
from idc import DatasetBuilder, UnsplashSource
from idc.filters import QualityFilter, Deduplicator
from idc.exporters import RawExporter

builder = DatasetBuilder("./animals")
builder.add_source(UnsplashSource(api_key="..."))
builder.add_filter(QualityFilter(min_width=512, min_height=512))
builder.add_filter(Deduplicator())

for cls in ["golden retriever", "tabby cat", "budgerigar"]:
    records = builder.search(cls, count=200)
    builder.download(records, label=cls.replace(" ", "_"))

builder.export(RawExporter())
```

### CLIP semantic filtering

```python
from idc.filters import CLIPFilter

# Requires: pip install "image-dataset-creator[clip]"
builder.add_filter(CLIPFilter(threshold=0.25))
```

Images whose cosine similarity to the search query falls below the threshold are rejected. The filter fails open — if CLIP is not installed or a file can't be scored, the image is accepted.

### Streaming export

All exporters accept any `Iterable[ImageRecord]`, so you can export directly from `iter_all()` without loading everything into memory:

```python
from idc.exporters import CSVExporter

exporter = CSVExporter()
exporter.export(builder.manifest.iter_all(), output_dir="./csv_export")
```

Split-based exporters (YOLO, COCO, HuggingFace) materialise records internally since they need a total count for proportional splitting.

### Save failure log

```python
accepted = builder.download(records, save_failure_log=True)
# Writes download_failures.jsonl to the dataset output directory
```

### Rate limiting

Avoid hitting API rate limits by adding a delay between paginated requests:

```python
from idc.sources import UnsplashSource, PexelsSource

builder.add_source(UnsplashSource(api_key="...", request_delay=0.5))  # 0.5 s between pages
builder.add_source(PexelsSource(api_key="...", request_delay=1.0))
```

### Per-page search progress

Use the `on_page` callback to track pagination progress in custom scripts:

```python
from idc.sources import UnsplashSource

source = UnsplashSource(api_key="...")
records = source.search("dogs", count=200, on_page=lambda n: print(f"Page {n}"))
```

The CLI uses this internally to render a live tqdm progress bar per source.

### Open Images (no API key)

```python
from idc.sources.openimages import OpenImagesSource

builder.add_source(OpenImagesSource(split="validation"))
records = builder.search("Dog", count=50)
```

The index is built from Open Images metadata CSVs on first use and cached locally at `~/.idc/openimages/`.

Available splits: `validation` (~41K images, default), `test` (~125K), `train` (~1.7M).

### Accessing the manifest

```python
# Load all records into memory at once
for record in builder.manifest.get_all():
    print(record.attribution)
    print(record.local_path)
    print(record.license_type)

# Stream records in batches (constant memory, good for large datasets)
for record in builder.manifest.iter_all(batch_size=500):
    print(record.source_id)
```

---

## Supported Sources

| Source | License | API Key | Notes |
|---|---|---|---|
| **Unsplash** | Unsplash License | Required (free) | High quality photography |
| **Pexels** | Pexels License | Required (free) | Curated, clean images |
| **Pixabay** | Pixabay License | Required (free) | Large volume, mixed quality |
| **Wikimedia** | CC0, CC-BY-4.0 | None | Wide variety; attribution may be required |
| **Open Images** | CC-BY-2.0 | None | Google dataset; local SQLite index built on first use |

All sources are **commercial-use OK** by default.

Wikimedia CC-BY-SA is excluded by default (copyleft); opt in with:

```python
from idc.sources import WikimediaSource
WikimediaSource(include_cc_by_sa=True)
```

---

## License Compliance

Every downloaded image has its license and attribution stored in `manifest.db`. Run `idc report` to generate a CSV for legal review.

Images with unknown or non-commercial licenses are **never downloaded** — the pipeline enforces this at the source adapter level.

---

## Development

```bash
git clone https://github.com/yourname/image-dataset-creator
cd image-dataset-creator
pip install -e ".[dev]"
pytest
```

### Running tests

```bash
# All tests with coverage report
pytest

# Fast (no coverage)
pytest --no-cov

# Single module
pytest tests/test_sources.py
```

### Linting

```bash
pip install ruff
ruff check idc/ tests/
```

### CI

GitHub Actions runs the full test matrix (Python 3.10–3.13) on every push and pull request. See [`.github/workflows/test.yml`](.github/workflows/test.yml).
