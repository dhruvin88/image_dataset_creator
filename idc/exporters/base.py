from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import ImageRecord


class BaseExporter(ABC):
    @abstractmethod
    def export(self, records: List[ImageRecord], output_dir: Path) -> None:
        """Export records to output_dir in the target format."""
        ...
