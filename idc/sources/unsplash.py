from __future__ import annotations

import time
from typing import Callable, List, Optional

import httpx

from ..models import ImageRecord
from ..utils import retry_request
from .base import ImageSource


class UnsplashSource(ImageSource):
    """Unsplash image source (Unsplash License — commercial OK)."""

    name = "unsplash"
    _BASE = "https://api.unsplash.com"
    _PER_PAGE = 30  # Unsplash max per request

    def __init__(self, api_key: str, request_delay: float = 0.0) -> None:
        self.api_key = api_key
        self._headers = {"Authorization": f"Client-ID {api_key}"}
        self.request_delay = request_delay

    def search(self, query: str, count: int, *, on_page: Optional[Callable[[int], None]] = None, **kwargs) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        page = 1
        page_count = 0

        with httpx.Client(timeout=30, headers=self._headers) as client:
            while len(records) < count:
                needed = count - len(records)
                resp = retry_request(
                    client,
                    "GET",
                    f"{self._BASE}/search/photos",
                    params={
                        "query": query,
                        "per_page": min(self._PER_PAGE, needed),
                        "page": page,
                    },
                )
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                page_results = 0
                for item in results:
                    records.append(self._parse(item, query))
                    page_results += 1
                    if len(records) >= count:
                        break

                page_count += page_results
                if on_page is not None:
                    on_page(page_results)

                total_pages = data.get("total_pages", 1)
                if page >= total_pages or len(results) < self._PER_PAGE:
                    break
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                page += 1

        return records

    def _parse(self, item: dict, query: str) -> ImageRecord:
        user = item.get("user", {})
        urls = item.get("urls", {})
        return ImageRecord(
            source="unsplash",
            source_id=item["id"],
            url=item.get("links", {}).get("html", ""),
            download_url=urls.get("regular", urls.get("full", "")),
            license_type="unsplash",
            license_url="https://unsplash.com/license",
            photographer=user.get("name", ""),
            photographer_url=user.get("links", {}).get("html", ""),
            attribution=f"Photo by {user.get('name', 'Unknown')} on Unsplash",
            width=item.get("width", 0),
            height=item.get("height", 0),
            tags=[t["title"] for t in item.get("tags", []) if isinstance(t, dict) and "title" in t],
            description=item.get("description") or item.get("alt_description") or "",
            query=query,
        )
