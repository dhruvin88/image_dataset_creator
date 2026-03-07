from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import yaml

from .base import BaseExporter
from ..models import ImageRecord
from ..utils import split_records


class YOLOExporter(BaseExporter):
    """
    Export as YOLO dataset structure.

    Directory layout:
        images/train/, images/val/, images/test/   (test only if test_split > 0)
        labels/train/, labels/val/, labels/test/
        dataset.yaml
    Labels are empty .txt files (unlabeled — ready for annotation).
    """

    def __init__(
        self,
        val_split: float = 0.1,
        test_split: float = 0.0,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.val_split = val_split
        self.test_split = test_split
        self.class_names = class_names or ["object"]

    def export(self, records: List[ImageRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        train, val, test = split_records(records, self.val_split, self.test_split)
        splits: dict[str, List[ImageRecord]] = {"train": train, "val": val}
        if self.test_split > 0:
            splits["test"] = test

        for split_name in splits:
            (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for split_name, split_recs in splits.items():
            valid = [r for r in split_recs if r.local_path and Path(r.local_path).exists()]
            for record in valid:
                src = Path(record.local_path)  # type: ignore[arg-type]
                shutil.copy2(src, output_dir / "images" / split_name / src.name)
                (output_dir / "labels" / split_name / (src.stem + ".txt")).touch()

        dataset_yaml: dict = {
            "path": str(output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        if self.test_split > 0:
            dataset_yaml["test"] = "images/test"

        with open(output_dir / "dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
