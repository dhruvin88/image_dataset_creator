"""Tests for ImageRecord model."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from idc.models import ImageRecord


def make_record(**overrides) -> ImageRecord:
    defaults = dict(
        source="pexels",
        source_id="999",
        url="https://pexels.com/photo/999",
        download_url="https://images.pexels.com/photos/999/photo.jpg",
        license_type="pexels",
        license_url="https://www.pexels.com/license/",
        attribution="Photo by Bob on Pexels",
        photographer="Bob",
        photographer_url="https://pexels.com/@bob",
        width=1920,
        height=1080,
        query="cats",
    )
    defaults.update(overrides)
    return ImageRecord(**defaults)


class TestImageRecord:
    def test_default_id_generated(self):
        r = make_record()
        assert len(r.id) == 36  # UUID format

    def test_unique_ids(self):
        r1, r2 = make_record(), make_record()
        assert r1.id != r2.id

    def test_default_downloaded_at(self):
        r = make_record()
        assert isinstance(r.downloaded_at, datetime)

    def test_optional_fields_default_none(self):
        r = make_record()
        assert r.local_path is None
        assert r.blur_score is None
        assert r.phash is None

    def test_tags_default_empty_list(self):
        r = make_record()
        assert r.tags == []

    def test_to_dict_roundtrip(self):
        r = make_record()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "pexels"
        assert d["local_path"] is None
        assert isinstance(d["downloaded_at"], str)  # ISO string

        restored = ImageRecord.from_dict(d)
        assert restored.source == r.source
        assert restored.source_id == r.source_id
        assert restored.downloaded_at == r.downloaded_at

    def test_to_dict_with_local_path(self, tmp_path):
        r = make_record()
        r.local_path = tmp_path / "img.jpg"
        d = r.to_dict()
        assert isinstance(d["local_path"], str)
        assert d["local_path"] == str(tmp_path / "img.jpg")

    def test_from_dict_with_path_string(self, tmp_path):
        r = make_record()
        r.local_path = tmp_path / "img.jpg"
        d = r.to_dict()
        restored = ImageRecord.from_dict(d)
        assert restored.local_path == Path(str(tmp_path / "img.jpg"))

    def test_json_serializable(self):
        r = make_record()
        d = r.to_dict()
        # Should not raise
        json.dumps(d)
