from __future__ import annotations

import re
from typing import List, Optional, Tuple

import httpx

from ..models import ImageRecord
from ..utils import retry_request
from .base import ImageSource

_LICENSE_MAP: dict[str, Tuple[str, str]] = {
    "cc0": ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "public domain": ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    # cc-by-sa MUST come before cc-by to avoid substring false-positive
    "cc-by-sa": ("CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    "cc-by": ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"),
}

_DEFAULT_ALLOWED = {"cc0", "public domain", "cc-by"}


class WikimediaSource(ImageSource):
    """Wikimedia Commons image source (CC0, CC-BY, and optionally CC-BY-SA)."""

    name = "wikimedia"
    _API = "https://commons.wikimedia.org/w/api.php"
    _SEARCH_LIMIT = 50

    def __init__(self, include_cc_by_sa: bool = False) -> None:
        self._allowed = set(_DEFAULT_ALLOWED)
        if include_cc_by_sa:
            self._allowed.add("cc-by-sa")

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        offset = 0

        with httpx.Client(timeout=30) as client:
            while len(records) < count:
                needed = count - len(records)
                search_results = self._search_files(client, query, min(self._SEARCH_LIMIT, needed * 2), offset)
                if not search_results:
                    break

                titles = "|".join(r["title"] for r in search_results)
                pages = self._get_image_info(client, titles)

                for page in pages.values():
                    record = self._parse(page, query)
                    if record is not None:
                        records.append(record)
                    if len(records) >= count:
                        break

                offset += len(search_results)
                if len(search_results) < self._SEARCH_LIMIT:
                    break

        return records

    def _search_files(self, client: httpx.Client, query: str, limit: int, offset: int) -> list:
        resp = retry_request(
            client,
            "GET",
            self._API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": f"{query} filetype:bitmap",
                "srnamespace": "6",
                "srlimit": limit,
                "sroffset": offset,
                "format": "json",
            },
        )
        return resp.json().get("query", {}).get("search", [])

    def _get_image_info(self, client: httpx.Client, titles: str) -> dict:
        resp = retry_request(
            client,
            "GET",
            self._API,
            params={
                "action": "query",
                "titles": titles,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mime",
                "format": "json",
            },
        )
        return resp.json().get("query", {}).get("pages", {})

    def _resolve_license(self, extmetadata: dict) -> Optional[Tuple[str, str]]:
        raw = extmetadata.get("LicenseShortName", {}).get("value", "").lower()
        for key, val in _LICENSE_MAP.items():
            if key in raw:
                # Exact key match against allowed set (avoids "cc-by" ⊂ "cc-by-sa" false positive)
                return val if key in self._allowed else None
        return None

    def _parse(self, page: dict, query: str) -> Optional[ImageRecord]:
        if str(page.get("pageid", "-1")) == "-1":
            return None

        imageinfo = page.get("imageinfo", [{}])
        if not imageinfo:
            return None
        info = imageinfo[0]

        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            return None

        extmetadata = info.get("extmetadata", {})
        result = self._resolve_license(extmetadata)
        if result is None:
            return None
        license_type, license_url = result

        artist_raw = extmetadata.get("Artist", {}).get("value", "")
        artist = re.sub(r"<[^>]+>", "", artist_raw).strip()
        attribution = f"Image by {artist} via Wikimedia Commons" if artist else "Image via Wikimedia Commons"
        title = page.get("title", "").replace("File:", "")

        return ImageRecord(
            source="wikimedia",
            source_id=str(page.get("pageid", "")),
            url=f"https://commons.wikimedia.org/wiki/{page.get('title', '').replace(' ', '_')}",
            download_url=info.get("url", ""),
            license_type=license_type,
            license_url=license_url,
            photographer=artist,
            photographer_url="",
            attribution=attribution,
            width=info.get("width", 0),
            height=info.get("height", 0),
            tags=[],
            description=title,
            query=query,
        )
