from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List

from .base import BaseExporter
from ..models import ImageRecord
from ..utils import split_records


class RawExporter(BaseExporter):
    """
    Export as flat images/ folder + metadata.jsonl.

    Without splits, records are streamed one at a time.
    With splits, the iterable is materialised once to compute split boundaries.
    """

    def __init__(self, val_split: float = 0.0, test_split: float = 0.0) -> None:
        self.val_split = val_split
        self.test_split = test_split

    def export(self, records: Iterable[ImageRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        use_splits = self.val_split > 0 or self.test_split > 0

        if use_splits:
            record_list: List[ImageRecord] = list(records)
            self._export_with_splits(record_list, output_dir)
        else:
            self._export_streaming(records, output_dir)

    def _export_streaming(self, records: Iterable[ImageRecord], output_dir: Path) -> None:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.jsonl"

        with open(metadata_path, "w") as f:
            for record in records:
                if not record.local_path:
                    continue
                src = Path(record.local_path)
                if not src.exists():
                    continue
                dest = images_dir / src.name
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                d = record.to_dict()
                d["local_path"] = str(dest.relative_to(output_dir))
                f.write(json.dumps(d) + "\n")

    def _export_with_splits(self, records: List[ImageRecord], output_dir: Path) -> None:
        train, val, test = split_records(records, self.val_split, self.test_split)
        splits = [("train", train), ("val", val), ("test", test)]
        metadata_lines: List[str] = []

        for split_name, split_recs in splits:
            images_dir = output_dir / "images" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)

            for record in split_recs:
                if not record.local_path:
                    continue
                src = Path(record.local_path)
                if not src.exists():
                    continue
                dest = images_dir / src.name
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                d = record.to_dict()
                d["local_path"] = str(dest.relative_to(output_dir))
                d["split"] = split_name
                metadata_lines.append(json.dumps(d))

        if metadata_lines:
            with open(output_dir / "metadata.jsonl", "w") as f:
                f.write("\n".join(metadata_lines) + "\n")
