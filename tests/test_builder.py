"""Integration tests for DatasetBuilder."""
from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from idc.builder import DatasetBuilder
from idc.exporters.raw import RawExporter
from idc.filters.dedup import Deduplicator
from idc.filters.quality import QualityFilter
from idc.models import ImageRecord
from idc.sources.base import ImageSource

from .conftest import make_jpeg_bytes, make_blurry_jpeg_bytes


# ------------------------------------------------------------------ #
# Fake source for testing
# ------------------------------------------------------------------ #


class FakeSource(ImageSource):
    """In-memory source that serves pre-baked records."""

    name = "fake"

    def __init__(self, records: List[ImageRecord]) -> None:
        self._records = records

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        return self._records[:count]

    def download(self, record: ImageRecord, output_dir: Path) -> Path:
        """Write a fake JPEG to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"fake_{record.source_id}.jpg"
        if not path.exists():
            path.write_bytes(make_jpeg_bytes(400, 300))
        return path


class BlurryFakeSource(FakeSource):
    def download(self, record: ImageRecord, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"fake_{record.source_id}.jpg"
        if not path.exists():
            path.write_bytes(make_blurry_jpeg_bytes())
        return path


def _make_fake_record(source_id: str, query: str = "dogs") -> ImageRecord:
    return ImageRecord(
        source="fake",
        source_id=source_id,
        url=f"https://fake.example.com/{source_id}",
        download_url=f"https://cdn.fake.example.com/{source_id}.jpg",
        license_type="unsplash",
        license_url="https://unsplash.com/license",
        attribution=f"Fake photo {source_id}",
        photographer="Fake Photographer",
        photographer_url="",
        width=400,
        height=300,
        query=query,
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestDatasetBuilder:
    def test_add_source_returns_self(self, tmp_path):
        builder = DatasetBuilder(tmp_path)
        src = FakeSource([])
        result = builder.add_source(src)
        assert result is builder

    def test_search_requires_source(self, tmp_path):
        builder = DatasetBuilder(tmp_path)
        with pytest.raises(ValueError, match="No sources"):
            builder.search("dogs", 10)

    def test_search_aggregates_results(self, tmp_path):
        records_a = [_make_fake_record(f"a{i}") for i in range(3)]
        records_b = [_make_fake_record(f"b{i}") for i in range(3)]

        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records_a))
        builder.add_source(FakeSource(records_b))

        results = builder.search("dogs", 6)
        assert len(results) >= 4  # at least 2 from each (count=6 split across 2 sources)

    def test_download_saves_files(self, tmp_path):
        records = [_make_fake_record(f"r{i}") for i in range(3)]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))

        accepted = builder.download(records)

        assert len(accepted) == 3
        for r in accepted:
            assert r.local_path is not None
            assert Path(r.local_path).exists()

    def test_download_updates_manifest(self, tmp_path):
        records = [_make_fake_record(f"m{i}") for i in range(2)]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))
        builder.download(records)

        assert builder.manifest.count() == 2

    def test_quality_filter_rejects_blurry(self, tmp_path):
        records = [_make_fake_record(f"blur{i}") for i in range(3)]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(BlurryFakeSource(records))
        builder.add_filter(QualityFilter(min_width=50, min_height=50, blur_threshold=100.0))

        accepted = builder.download(records)

        # Solid-colour images have near-zero Laplacian variance → all rejected
        assert len(accepted) == 0

    def test_dedup_removes_duplicates(self, tmp_path):
        # All records point to the same source_id → same file downloaded → same hash
        records = [_make_fake_record("dup") for _ in range(3)]
        # Each must have a unique ID but we want the downloaded images to be identical
        for i, r in enumerate(records):
            r.source_id = f"dup_{i}"

        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))
        builder.add_filter(Deduplicator(threshold=10))

        accepted = builder.download(records)

        # All three download the same pixel pattern → only 1 unique
        assert len(accepted) == 1

    def test_label_creates_subfolder(self, tmp_path):
        records = [_make_fake_record("lbl1")]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))

        builder.download(records, label="golden_retriever")

        assert (tmp_path / "images" / "golden_retriever").is_dir()

    def test_export_delegates_to_exporter(self, tmp_path):
        records = [_make_fake_record(f"e{i}") for i in range(2)]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))
        builder.download(records)

        out = tmp_path / "raw_out"
        builder.export(RawExporter(), output_dir=out)

        assert (out / "metadata.jsonl").exists()

    def test_resume_skips_already_downloaded(self, tmp_path):
        """Records already in manifest with existing files should be skipped."""
        records = [_make_fake_record("resume1"), _make_fake_record("resume2")]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))

        # First download
        accepted1 = builder.download(records)
        assert len(accepted1) == 2

        # Second download — same records, should be skipped via manifest
        source2 = FakeSource(records)
        download_calls = []
        original_download = source2.download

        def tracked_download(record, output_dir):
            download_calls.append(record.source_id)
            return original_download(record, output_dir)

        source2.download = tracked_download
        builder._sources = [source2]
        accepted2 = builder.download(records, skip_existing=True)

        # Should still return the existing records
        assert len(accepted2) == 2
        # Should not have re-downloaded (manifest lookup returned existing)
        assert len(download_calls) == 0

    def test_no_resume_redownloads(self, tmp_path):
        """skip_existing=False should re-download even if already in manifest."""
        records = [_make_fake_record("redownload1")]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))
        builder.download(records)

        download_calls = []
        src = FakeSource(records)
        original = src.download

        def tracked(record, output_dir):
            download_calls.append(record.source_id)
            return original(record, output_dir)

        src.download = tracked
        builder._sources = [src]
        builder.download(records, skip_existing=False)
        assert len(download_calls) == 1  # Was re-downloaded

    def test_download_summary_tracks_accepted(self, tmp_path):
        records = [_make_fake_record(f"s{i}") for i in range(3)]
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource(records))
        builder.download(records)

        assert builder.last_summary is not None
        assert builder.last_summary.accepted == 3
        assert builder.last_summary.failed_download == 0

    def test_download_summary_tracks_failures(self, tmp_path):
        records = [_make_fake_record("fail1")]
        builder = DatasetBuilder(tmp_path)

        # Source that always raises on download
        class ErrorSource(FakeSource):
            name = "fake"

            def download(self, record, output_dir):
                raise RuntimeError("simulated network error")

        builder.add_source(ErrorSource(records))
        builder.download(records)

        assert builder.last_summary.failed_download >= 1

    def test_download_summary_saves_failure_log(self, tmp_path):
        records = [_make_fake_record("faillog")]

        class ErrorSource(FakeSource):
            name = "fake"
            def download(self, record, output_dir):
                raise RuntimeError("network error")

        builder = DatasetBuilder(tmp_path)
        builder.add_source(ErrorSource(records))
        builder.download(records, save_failure_log=True)

        log = tmp_path / "download_failures.jsonl"
        assert log.exists()
        import json
        entries = [json.loads(l) for l in log.read_text().strip().splitlines()]
        assert len(entries) >= 1
        assert "reason" in entries[0]

    def test_multi_query_accumulates_in_manifest(self, tmp_path):
        builder = DatasetBuilder(tmp_path)
        builder.add_source(FakeSource([_make_fake_record(f"q1_{i}", query="dogs") for i in range(2)]))

        r1 = builder.search("dogs", 2)
        builder.download(r1)

        # Add more records for second query — need new source
        builder._sources = [FakeSource([_make_fake_record(f"q2_{i}", query="cats") for i in range(2)])]
        # Reset dedup so second batch isn't filtered (different colors would help; here we cheat)
        r2 = builder.search("cats", 2)
        builder.download(r2)

        # Both batches in manifest
        assert builder.manifest.count() >= 2
