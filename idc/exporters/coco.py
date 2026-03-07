from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from .base import BaseExporter
from ..models import ImageRecord
from ..utils import split_records


class COCOExporter(BaseExporter):
    """
    Export as COCO-format dataset.

    Without splits:
        images/
        annotations/instances.json

    With splits (val_split or test_split > 0):
        images/train/, images/val/, images/test/
        annotations/train.json, annotations/val.json, annotations/test.json
    """

    def __init__(
        self,
        dataset_name: str = "idc_dataset",
        val_split: float = 0.0,
        test_split: float = 0.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.val_split = val_split
        self.test_split = test_split

    def export(self, records: List[ImageRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        use_splits = self.val_split > 0 or self.test_split > 0

        if use_splits:
            train, val, test = split_records(records, self.val_split, self.test_split)
            splits = [("train", train), ("val", val)]
            if self.test_split > 0:
                splits.append(("test", test))
        else:
            splits = [("", records)]

        for split_name, split_recs in splits:
            images_dir = output_dir / "images" / split_name if split_name else output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            coco_images = []
            for img_id, record in enumerate(split_recs, start=1):
                if not record.local_path:
                    continue
                src = Path(record.local_path)
                if not src.exists():
                    continue

                dest = images_dir / src.name
                if src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)

                coco_images.append(
                    {
                        "id": img_id,
                        "file_name": src.name,
                        "width": record.width,
                        "height": record.height,
                        "license": record.license_type,
                        "attribution": record.attribution,
                        "url": record.url,
                        "idc_id": record.id,
                        "source": record.source,
                    }
                )

            coco_data = {
                "info": {
                    "description": self.dataset_name,
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().isoformat(),
                },
                "licenses": [],
                "images": coco_images,
                "annotations": [],
                "categories": [],
            }

            annotation_file = split_name + ".json" if split_name else "instances.json"
            with open(annotations_dir / annotation_file, "w") as f:
                json.dump(coco_data, f, indent=2)
