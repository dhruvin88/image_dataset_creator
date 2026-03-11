from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Optional

from .base import BaseExporter
from ..models import ImageRecord


class CSVExporter(BaseExporter):
    """
    Export dataset as a flat CSV file.

    Records are written one at a time so the full dataset never needs to fit
    in memory.  Output: ``dataset.csv`` with one row per image.
    """

    def __init__(self, filename: str = "dataset.csv") -> None:
        self.filename = filename

    def export(self, records: Iterable[ImageRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / self.filename

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer: Optional[csv.DictWriter] = None
            for record in records:
                row = record.to_dict()
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                writer.writerow(row)

        # Remove the file if nothing was written (empty records)
        if writer is None:
            out_path.unlink(missing_ok=True)
