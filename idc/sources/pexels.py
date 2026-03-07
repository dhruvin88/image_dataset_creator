from __future__ import annotations

from typing import List

import httpx

from ..models import ImageRecord
from ..utils import retry_request
from .base import ImageSource


class PexelsSource(ImageSource):
    """Pexels image source (Pexels License — commercial OK)."""

    name = "pexels"
    _BASE = "https://api.pexels.com/v1"
    _PER_PAGE = 80  # Pexels max per request

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._headers = {"Authorization": api_key}

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        page = 1

        with httpx.Client(timeout=30, headers=self._headers) as client:
            while len(records) < count:
                needed = count - len(records)
                resp = retry_request(
                    client,
                    "GET",
                    f"{self._BASE}/search",
                    params={
                        "query": query,
                        "per_page": min(self._PER_PAGE, needed),
                        "page": page,
                    },
                )
                data = resp.json()
                photos = data.get("photos", [])
                if not photos:
                    break

                for photo in photos:
                    records.append(self._parse(photo, query))
                    if len(records) >= count:
                        break

                if not data.get("next_page") or len(photos) < self._PER_PAGE:
                    break
                page += 1

        return records

    def _parse(self, photo: dict, query: str) -> ImageRecord:
        src = photo.get("src", {})
        return ImageRecord(
            source="pexels",
            source_id=str(photo["id"]),
            url=photo.get("url", ""),
            download_url=src.get("original", src.get("large2x", "")),
            license_type="pexels",
            license_url="https://www.pexels.com/license/",
            photographer=photo.get("photographer", ""),
            photographer_url=photo.get("photographer_url", ""),
            attribution=f"Photo by {photo.get('photographer', 'Unknown')} on Pexels",
            width=photo.get("width", 0),
            height=photo.get("height", 0),
            tags=[],
            description=photo.get("alt", ""),
            query=query,
        )
