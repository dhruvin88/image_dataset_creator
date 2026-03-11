"""Tests for Manifest SQLite storage."""
from __future__ import annotations

import pytest

from idc.manifest import Manifest
from idc.models import ImageRecord


def _make_record(source: str = "unsplash", source_id: str = "id1", query: str = "dogs") -> ImageRecord:
    return ImageRecord(
        source=source,
        source_id=source_id,
        url=f"https://example.com/{source_id}",
        download_url=f"https://cdn.example.com/{source_id}.jpg",
        license_type="unsplash",
        license_url="https://unsplash.com/license",
        attribution="Photo by Test on Unsplash",
        photographer="Test",
        photographer_url="https://unsplash.com/@test",
        width=800,
        height=600,
        query=query,
    )


class TestManifest:
    def test_add_and_get_all(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record()
        db.add(r)
        records = db.get_all()
        assert len(records) == 1
        assert records[0].source_id == "id1"

    def test_count(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        assert db.count() == 0
        db.add(_make_record(source_id="a"))
        db.add(_make_record(source_id="b"))
        assert db.count() == 2

    def test_upsert_on_same_id(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record()
        db.add(r)
        # Modify and re-add with same id
        r.description = "updated"
        db.add(r)
        assert db.count() == 1
        assert db.get_all()[0].description == "updated"

    def test_get_phashes(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record()
        r.phash = "aabbccdd"
        db.add(r)

        r2 = _make_record(source_id="id2")
        r2.phash = "11223344"
        db.add(r2)

        phashes = db.get_phashes()
        assert set(phashes) == {"aabbccdd", "11223344"}

    def test_get_by_source(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        db.add(_make_record(source="unsplash", source_id="u1"))
        db.add(_make_record(source="pexels", source_id="p1"))
        db.add(_make_record(source="pexels", source_id="p2"))

        assert len(db.get_by_source("pexels")) == 2
        assert len(db.get_by_source("unsplash")) == 1
        assert len(db.get_by_source("pixabay")) == 0

    def test_get_by_id(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record()
        db.add(r)
        found = db.get_by_id(r.id)
        assert found is not None
        assert found.id == r.id

    def test_get_by_id_missing_returns_none(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        assert db.get_by_id("nonexistent-id") is None

    def test_records_survive_reopen(self, tmp_path):
        db_path = tmp_path / "manifest.db"
        db = Manifest(db_path)
        r = _make_record()
        db.add(r)
        del db

        db2 = Manifest(db_path)
        assert db2.count() == 1
        assert db2.get_all()[0].source_id == "id1"

    def test_remove_single(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record()
        db.add(r)
        assert db.count() == 1
        db.remove(r.id)
        assert db.count() == 0

    def test_remove_many(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        records = [_make_record(source_id=f"r{i}") for i in range(5)]
        for r in records:
            db.add(r)
        assert db.count() == 5

        ids_to_remove = [records[0].id, records[2].id, records[4].id]
        removed = db.remove_many(ids_to_remove)
        assert removed == 3
        assert db.count() == 2

    def test_remove_many_empty_list(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        db.add(_make_record())
        result = db.remove_many([])
        assert result == 0
        assert db.count() == 1

    def test_has_source_id_true(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        db.add(_make_record(source="pexels", source_id="px999"))
        assert db.has_source_id("pexels", "px999") is True

    def test_has_source_id_false(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        db.add(_make_record(source="pexels", source_id="px999"))
        assert db.has_source_id("pexels", "px000") is False
        assert db.has_source_id("unsplash", "px999") is False  # wrong source

    def test_get_by_source_id(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        r = _make_record(source="pixabay", source_id="pb42")
        db.add(r)
        found = db.get_by_source_id("pixabay", "pb42")
        assert found is not None
        assert found.source_id == "pb42"

    def test_get_by_source_id_not_found(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        assert db.get_by_source_id("unsplash", "nope") is None

    def test_iter_all_empty(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        assert list(db.iter_all()) == []

    def test_iter_all_yields_all_records(self, tmp_path):
        db = Manifest(tmp_path / "manifest.db")
        records = [_make_record(source_id=f"r{i}") for i in range(5)]
        for r in records:
            db.add(r)
        result = list(db.iter_all())
        assert len(result) == 5
        assert {r.source_id for r in result} == {f"r{i}" for i in range(5)}

    def test_iter_all_small_batch_size(self, tmp_path):
        """batch_size smaller than record count forces multiple LIMIT/OFFSET queries."""
        db = Manifest(tmp_path / "manifest.db")
        for i in range(7):
            db.add(_make_record(source_id=f"b{i}"))
        result = list(db.iter_all(batch_size=3))
        assert len(result) == 7

    def test_iter_all_exact_batch_boundary(self, tmp_path):
        """Exactly batch_size records causes one extra (empty) query."""
        db = Manifest(tmp_path / "manifest.db")
        for i in range(4):
            db.add(_make_record(source_id=f"e{i}"))
        result = list(db.iter_all(batch_size=4))
        assert len(result) == 4
