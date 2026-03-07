from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import httpx

from ..models import ImageRecord
from ..utils import retry_request


class ImageSource(ABC):
    """Abstract base class for all image sources."""

    name: str = ""

    @abstractmethod
    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        """Search for images matching query. Returns up to count ImageRecords."""
        ...

    def download(self, record: ImageRecord, output_dir: Path) -> Path:
        """Download the image to output_dir. Returns local path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = self._guess_extension(record.download_url, record.format)
        filename = f"{record.source}_{record.source_id}{ext}"
        local_path = output_dir / filename

        if local_path.exists():
            return local_path

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = retry_request(client, "GET", record.download_url)
            local_path.write_bytes(resp.content)

        return local_path

    def _guess_extension(self, url: str, fmt: str = "") -> str:
        fmt_lower = fmt.lower()
        if fmt_lower in ("jpeg", "jpg"):
            return ".jpg"
        if fmt_lower == "png":
            return ".png"
        if fmt_lower == "webp":
            return ".webp"

        url_path = url.split("?")[0].split("/")[-1]
        if "." in url_path:
            ext = "." + url_path.rsplit(".", 1)[-1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                return ".jpg" if ext == ".jpeg" else ext

        return ".jpg"
