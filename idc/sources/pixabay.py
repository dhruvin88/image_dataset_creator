from __future__ import annotations

from typing import List

import httpx

from ..models import ImageRecord
from ..utils import retry_request
from .base import ImageSource


class PixabaySource(ImageSource):
    """Pixabay image source (Pixabay License — commercial OK, no attribution required)."""

    name = "pixabay"
    _BASE = "https://pixabay.com/api/"
    _PER_PAGE = 200  # Pixabay max per request

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        page = 1

        with httpx.Client(timeout=30) as client:
            while len(records) < count:
                needed = count - len(records)
                resp = retry_request(
                    client,
                    "GET",
                    self._BASE,
                    params={
                        "key": self.api_key,
                        "q": query,
                        "per_page": min(self._PER_PAGE, max(3, needed)),
                        "page": page,
                        "image_type": "photo",
                        "safesearch": "true",
                    },
                )
                data = resp.json()
                hits = data.get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    records.append(self._parse(hit, query))
                    if len(records) >= count:
                        break

                total_hits = data.get("totalHits", 0)
                if len(records) >= total_hits or len(hits) < 3:
                    break
                page += 1

        return records

    def _parse(self, hit: dict, query: str) -> ImageRecord:
        user = hit.get("user", "Unknown")
        user_id = hit.get("user_id", "")
        tags = [t.strip() for t in hit.get("tags", "").split(",") if t.strip()]
        return ImageRecord(
            source="pixabay",
            source_id=str(hit["id"]),
            url=hit.get("pageURL", ""),
            download_url=hit.get("largeImageURL", hit.get("webformatURL", "")),
            license_type="pixabay",
            license_url="https://pixabay.com/service/license-summary/",
            photographer=user,
            photographer_url=f"https://pixabay.com/users/{user}-{user_id}/",
            attribution=f"Image by {user} on Pixabay",
            width=hit.get("imageWidth", 0),
            height=hit.get("imageHeight", 0),
            tags=tags,
            description=hit.get("tags", ""),
            query=query,
        )
