from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import imagehash
from PIL import Image

from ..models import ImageRecord


class Deduplicator:
    """
    Perceptual hash-based deduplication.

    threshold: maximum Hamming distance to consider two images as duplicates.
    Lower = stricter (must be more similar to count as duplicate).
    Default 10: images with pHash distance <= 10 are considered duplicates.
    """

    def __init__(self, threshold: int = 10) -> None:
        self.threshold = threshold
        self._seen: List[imagehash.ImageHash] = []

    def load_existing(self, phashes: List[str]) -> None:
        """Seed the seen-hashes set from existing manifest data."""
        for h in phashes:
            try:
                self._seen.append(imagehash.hex_to_hash(h))
            except Exception:
                pass

    def compute_hash(self, image_path: Path) -> Optional[str]:
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception:
            return None

    def is_duplicate(self, phash_str: str) -> bool:
        try:
            h = imagehash.hex_to_hash(phash_str)
        except Exception:
            return False
        return any((h - seen) <= self.threshold for seen in self._seen)

    def add_hash(self, phash_str: str) -> None:
        try:
            self._seen.append(imagehash.hex_to_hash(phash_str))
        except Exception:
            pass

    def check_and_add(self, record: ImageRecord) -> Tuple[bool, str]:
        """
        Check if the record is a duplicate of a previously seen image.
        Atomically adds its hash if unique. Returns (is_unique, reason).
        """
        phash_str = record.phash
        if phash_str is None and record.local_path:
            phash_str = self.compute_hash(Path(record.local_path))

        if phash_str is None:
            return True, ""  # Cannot compute hash — allow through

        if self.is_duplicate(phash_str):
            return False, f"near-duplicate detected (phash={phash_str})"

        self.add_hash(phash_str)
        return True, ""
