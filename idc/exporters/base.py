from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ..models import ImageRecord


class BaseExporter(ABC):
    @abstractmethod
    def export(self, records: Iterable[ImageRecord], output_dir: Path) -> None:
        """Export records to output_dir in the target format.

        ``records`` may be a list, generator, or any other iterable.
        Exporters that require random access (e.g. split-based formats) will
        materialise the iterable into a list internally.
        """
        ...
