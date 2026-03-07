from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BaseExporter
from ..models import ImageRecord
from ..utils import split_records


class HuggingFaceExporter(BaseExporter):
    """
    Export as a HuggingFace datasets-compatible format (Parquet/Arrow).

    Without splits: saves a single Dataset.
    With splits (val_split or test_split > 0): saves a DatasetDict with
    'train', 'validation', and optionally 'test' keys.

    Requires: pip install image-dataset-creator[huggingface]
    """

    def __init__(self, val_split: float = 0.0, test_split: float = 0.0) -> None:
        self.val_split = val_split
        self.test_split = test_split

    def export(self, records: List[ImageRecord], output_dir: Path) -> None:
        try:
            import datasets as hf_datasets
        except ImportError as exc:
            raise ImportError(
                "HuggingFace 'datasets' library is required for HF export.\n"
                "Install with: pip install image-dataset-creator[huggingface]"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)

        features = hf_datasets.Features(
            {
                "image_path": hf_datasets.Value("string"),
                "id": hf_datasets.Value("string"),
                "source": hf_datasets.Value("string"),
                "license_type": hf_datasets.Value("string"),
                "attribution": hf_datasets.Value("string"),
                "photographer": hf_datasets.Value("string"),
                "url": hf_datasets.Value("string"),
                "width": hf_datasets.Value("int32"),
                "height": hf_datasets.Value("int32"),
                "tags": hf_datasets.Sequence(hf_datasets.Value("string")),
                "description": hf_datasets.Value("string"),
                "query": hf_datasets.Value("string"),
            }
        )

        use_splits = self.val_split > 0 or self.test_split > 0

        if use_splits:
            train, val, test = split_records(records, self.val_split, self.test_split)
            split_map = {"train": train, "validation": val}
            if self.test_split > 0:
                split_map["test"] = test

            dataset_dict = hf_datasets.DatasetDict(
                {
                    name: hf_datasets.Dataset.from_dict(
                        self._to_dict(recs), features=features
                    )
                    for name, recs in split_map.items()
                }
            )
            dataset_dict.save_to_disk(str(output_dir))
        else:
            valid = [r for r in records if r.local_path and Path(r.local_path).exists()]
            dataset = hf_datasets.Dataset.from_dict(self._to_dict(valid), features=features)
            dataset.save_to_disk(str(output_dir))

    @staticmethod
    def _to_dict(records: List[ImageRecord]) -> dict:
        valid = [r for r in records if r.local_path and Path(r.local_path).exists()]
        return {
            "image_path": [str(r.local_path) for r in valid],
            "id": [r.id for r in valid],
            "source": [r.source for r in valid],
            "license_type": [r.license_type for r in valid],
            "attribution": [r.attribution for r in valid],
            "photographer": [r.photographer for r in valid],
            "url": [r.url for r in valid],
            "width": [r.width for r in valid],
            "height": [r.height for r in valid],
            "tags": [r.tags for r in valid],
            "description": [r.description for r in valid],
            "query": [r.query for r in valid],
        }
