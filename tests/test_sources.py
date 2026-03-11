"""Tests for image source adapters (mocked HTTP)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from idc.sources.pixabay import PixabaySource
from idc.sources.pexels import PexelsSource
from idc.sources.unsplash import UnsplashSource
from idc.sources.wikimedia import WikimediaSource


# ------------------------------------------------------------------ #
# Shared mock builder
# ------------------------------------------------------------------ #


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


# ------------------------------------------------------------------ #
# Unsplash
# ------------------------------------------------------------------ #

_UNSPLASH_PAGE = {
    "results": [
        {
            "id": "ph001",
            "width": 1920,
            "height": 1080,
            "description": "Golden retriever",
            "alt_description": "dog in park",
            "urls": {"regular": "https://images.unsplash.com/photo-ph001?w=1080", "full": "https://images.unsplash.com/photo-ph001"},
            "links": {"html": "https://unsplash.com/photos/ph001"},
            "user": {"name": "Alice Smith", "links": {"html": "https://unsplash.com/@alice"}},
            "tags": [{"title": "dog"}, {"title": "retriever"}],
        }
    ],
    "total": 1,
    "total_pages": 1,
}


class TestUnsplashSource:
    def test_search_returns_records(self):
        source = UnsplashSource(api_key="test_key")
        mock_resp = _mock_response(_UNSPLASH_PAGE)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = mock_resp

            records = source.search("golden retriever", count=1)

        assert len(records) == 1
        r = records[0]
        assert r.source == "unsplash"
        assert r.source_id == "ph001"
        assert r.license_type == "unsplash"
        assert r.photographer == "Alice Smith"
        assert "Alice Smith" in r.attribution
        assert r.width == 1920
        assert r.height == 1080
        assert "dog" in r.tags

    def test_search_stops_at_count(self):
        """Should not request more pages than needed."""
        source = UnsplashSource(api_key="test_key")
        multi_page = {
            "results": [_UNSPLASH_PAGE["results"][0]] * 5,
            "total": 100,
            "total_pages": 10,
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(multi_page)

            records = source.search("dogs", count=3)

        assert len(records) <= 3

    def test_sets_query_on_records(self):
        source = UnsplashSource(api_key="key")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(_UNSPLASH_PAGE)
            records = source.search("cats", count=1)

        assert records[0].query == "cats"


# ------------------------------------------------------------------ #
# Pexels
# ------------------------------------------------------------------ #

_PEXELS_PAGE = {
    "photos": [
        {
            "id": 42,
            "url": "https://www.pexels.com/photo/42/",
            "photographer": "Bob Jones",
            "photographer_url": "https://www.pexels.com/@bob",
            "width": 1600,
            "height": 900,
            "alt": "A labrador",
            "src": {
                "original": "https://images.pexels.com/photos/42/photo.jpeg",
                "large2x": "https://images.pexels.com/photos/42/photo.jpeg?w=1920",
            },
        }
    ],
    "next_page": None,
}


class TestPexelsSource:
    def test_search_returns_records(self):
        source = PexelsSource(api_key="pexels_key")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(_PEXELS_PAGE)
            records = source.search("labrador", count=1)

        assert len(records) == 1
        r = records[0]
        assert r.source == "pexels"
        assert r.source_id == "42"
        assert r.license_type == "pexels"
        assert r.photographer == "Bob Jones"
        assert "Bob Jones" in r.attribution

    def test_license_is_pexels(self):
        source = PexelsSource(api_key="key")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(_PEXELS_PAGE)
            records = source.search("dogs", count=1)

        # Must be a commercially-safe license
        assert records[0].license_type == "pexels"


# ------------------------------------------------------------------ #
# Pixabay
# ------------------------------------------------------------------ #

_PIXABAY_PAGE = {
    "hits": [
        {
            "id": 777,
            "pageURL": "https://pixabay.com/photos/777",
            "largeImageURL": "https://pixabay.com/get/large.jpg",
            "webformatURL": "https://pixabay.com/get/web.jpg",
            "imageWidth": 1280,
            "imageHeight": 720,
            "tags": "cat, kitten, feline",
            "user": "pixuser",
            "user_id": 42,
        }
    ],
    "totalHits": 1,
}


class TestPixabaySource:
    def test_search_returns_records(self):
        source = PixabaySource(api_key="pixabay_key")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(_PIXABAY_PAGE)
            records = source.search("cat", count=1)

        assert len(records) == 1
        r = records[0]
        assert r.source == "pixabay"
        assert r.source_id == "777"
        assert r.license_type == "pixabay"
        assert r.photographer == "pixuser"
        assert r.tags == ["cat", "kitten", "feline"]

    def test_license_is_pixabay(self):
        source = PixabaySource(api_key="key")
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.return_value = _mock_response(_PIXABAY_PAGE)
            records = source.search("cat", count=1)

        assert records[0].license_type == "pixabay"


# ------------------------------------------------------------------ #
# Wikimedia
# ------------------------------------------------------------------ #

_WIKI_SEARCH = {
    "query": {
        "search": [{"title": "File:Golden retriever.jpg", "pageid": 100}]
    }
}

_WIKI_INFO = {
    "query": {
        "pages": {
            "100": {
                "pageid": 100,
                "title": "File:Golden retriever.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/golden.jpg",
                        "width": 2000,
                        "height": 1500,
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/zero/1.0/"},
                            "Artist": {"value": "Jane Wikimedia"},
                        },
                    }
                ],
            }
        }
    }
}


class TestWikimediaSource:
    def test_search_returns_records(self):
        source = WikimediaSource()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.side_effect = [
                _mock_response(_WIKI_SEARCH),
                _mock_response(_WIKI_INFO),
            ]
            records = source.search("golden retriever", count=1)

        assert len(records) == 1
        r = records[0]
        assert r.source == "wikimedia"
        assert r.license_type == "CC0-1.0"
        assert "Wikimedia" in r.attribution

    def test_filters_non_cc_licenses_by_default(self):
        """CC-BY-SA should be excluded unless include_cc_by_sa=True."""
        wiki_info_cc_by_sa = {
            "query": {
                "pages": {
                    "200": {
                        "pageid": 200,
                        "title": "File:Test.jpg",
                        "imageinfo": [
                            {
                                "url": "https://upload.wikimedia.org/test.jpg",
                                "width": 800,
                                "height": 600,
                                "mime": "image/jpeg",
                                "extmetadata": {
                                    "LicenseShortName": {"value": "CC-BY-SA 4.0"},
                                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                    "Artist": {"value": "Someone"},
                                },
                            }
                        ],
                    }
                }
            }
        }
        search_resp = {
            "query": {"search": [{"title": "File:Test.jpg", "pageid": 200}]}
        }

        source = WikimediaSource(include_cc_by_sa=False)
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.side_effect = [
                _mock_response(search_resp),
                _mock_response(wiki_info_cc_by_sa),
            ]
            records = source.search("test", count=5)

        assert len(records) == 0

    def test_includes_cc_by_sa_when_opted_in(self):
        wiki_info_cc_by_sa = {
            "query": {
                "pages": {
                    "200": {
                        "pageid": 200,
                        "title": "File:Test.jpg",
                        "imageinfo": [
                            {
                                "url": "https://upload.wikimedia.org/test.jpg",
                                "width": 800,
                                "height": 600,
                                "mime": "image/jpeg",
                                "extmetadata": {
                                    "LicenseShortName": {"value": "CC-BY-SA 4.0"},
                                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                    "Artist": {"value": "Someone"},
                                },
                            }
                        ],
                    }
                }
            }
        }
        search_resp = {
            "query": {"search": [{"title": "File:Test.jpg", "pageid": 200}]}
        }

        source = WikimediaSource(include_cc_by_sa=True)
        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.request.side_effect = [
                _mock_response(search_resp),
                _mock_response(wiki_info_cc_by_sa),
            ]
            records = source.search("test", count=5)

        assert len(records) == 1
        assert records[0].license_type == "CC-BY-SA-4.0"


# ------------------------------------------------------------------ #
# Base source download
# ------------------------------------------------------------------ #


class TestBaseSourceDownload:
    def test_download_saves_file(self, tmp_path):
        from idc.sources.unsplash import UnsplashSource

        source = UnsplashSource(api_key="key")
        from idc.models import ImageRecord

        record = ImageRecord(
            source="unsplash",
            source_id="dltest",
            url="https://unsplash.com/photos/dltest",
            download_url="https://images.unsplash.com/photo-dltest",
            license_type="unsplash",
            license_url="https://unsplash.com/license",
            attribution="Photo by Test on Unsplash",
            photographer="Test",
            photographer_url="",
            width=800,
            height=600,
            query="dogs",
        )

        fake_bytes = b"\xff\xd8\xff" + b"\x00" * 1000  # minimal fake JPEG header

        with patch("httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            resp.content = fake_bytes
            resp.raise_for_status = MagicMock()
            mock_client.request.return_value = resp

            local_path = source.download(record, tmp_path)

        assert local_path.exists()
        assert local_path.read_bytes() == fake_bytes

    def test_download_skips_existing_file(self, tmp_path):
        from idc.sources.unsplash import UnsplashSource

        source = UnsplashSource(api_key="key")
        from idc.models import ImageRecord

        record = ImageRecord(
            source="unsplash", source_id="existing",
            url="", download_url="https://example.com/img.jpg",
            license_type="unsplash", license_url="", attribution="", photographer="",
            photographer_url="", width=100, height=100, query="",
        )

        existing_file = tmp_path / "unsplash_existing.jpg"
        existing_file.write_bytes(b"existing content")

        with patch("httpx.Client") as mock_client_cls:
            local_path = source.download(record, tmp_path)

        # Should not have called client.get since file already exists
        mock_client_cls.return_value.__enter__.return_value.request.assert_not_called()
        assert local_path == existing_file


# ------------------------------------------------------------------ #
# Base source async download (adownload)
# ------------------------------------------------------------------ #


class TestBaseSourceADownload:
    """Tests for the default adownload() method on ImageSource (via UnsplashSource)."""

    def _make_record(self, source_id="adltest"):
        from idc.models import ImageRecord
        return ImageRecord(
            source="unsplash",
            source_id=source_id,
            url=f"https://unsplash.com/photos/{source_id}",
            download_url=f"https://images.unsplash.com/photo-{source_id}",
            license_type="unsplash",
            license_url="https://unsplash.com/license",
            attribution="Photo by Test on Unsplash",
            photographer="Test",
            photographer_url="",
            width=800,
            height=600,
            query="dogs",
        )

    def test_adownload_saves_file(self, tmp_path):
        source = UnsplashSource(api_key="key")
        record = self._make_record()
        fake_bytes = b"\xff\xd8\xff" + b"\x00" * 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.headers = {}
        ok_resp.raise_for_status = MagicMock()
        ok_resp.content = fake_bytes

        async def _fake_request(*args, **kwargs):
            return ok_resp

        with patch.object(httpx.AsyncClient, "request", new=_fake_request):
            local_path = asyncio.run(source.adownload(record, tmp_path / "async_out"))

        assert local_path.exists()
        assert local_path.read_bytes() == fake_bytes

    def test_adownload_skips_existing_file(self, tmp_path):
        source = UnsplashSource(api_key="key")
        record = self._make_record(source_id="existing_async")

        out_dir = tmp_path / "async_out"
        out_dir.mkdir()
        existing = out_dir / "unsplash_existing_async.jpg"
        existing.write_bytes(b"cached content")

        async def _fake_request(*args, **kwargs):
            raise AssertionError("should not be called when file exists")

        with patch.object(httpx.AsyncClient, "request", new=_fake_request):
            local_path = asyncio.run(source.adownload(record, out_dir))

        assert local_path == existing
        assert local_path.read_bytes() == b"cached content"

    def test_adownload_raises_on_network_error(self, tmp_path):
        source = UnsplashSource(api_key="key")
        record = self._make_record()

        async def _fake_request(*args, **kwargs):
            raise httpx.ConnectError("simulated network error")

        with patch.object(httpx.AsyncClient, "request", new=_fake_request):
            with pytest.raises(httpx.ConnectError):
                asyncio.run(source.adownload(record, tmp_path / "err_out"))
