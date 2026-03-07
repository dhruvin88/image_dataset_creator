from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .base import BaseExporter
from ..models import ImageRecord


class CSVExporter(BaseExporter):
    """
    Export dataset as a flat CSV file.

    Output: dataset.csv with one row per image and all metadata columns.
    """

    def __init__(self, filename: str = "dataset.csv") -> None:
        self.filename = filename

    def export(self, records: List[ImageRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not records:
            return

        fieldnames = list(records[0].to_dict().keys())
        out_path = output_dir / self.filename

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())
