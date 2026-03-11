"""Tests for quality filter, CLIP filter, and deduplicator."""
from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from idc.filters.clip_filter import CLIPFilter
from idc.filters.dedup import Deduplicator
from idc.filters.quality import QualityFilter
from idc.models import ImageRecord

from .conftest import make_blurry_jpeg_bytes, make_jpeg_bytes, make_sharp_jpeg_bytes


def _record_with_file(tmp_path: Path, img_bytes: bytes, source_id: str = "test") -> ImageRecord:
    path = tmp_path / f"{source_id}.jpg"
    path.write_bytes(img_bytes)
    return ImageRecord(
        source="unsplash",
        source_id=source_id,
        url="https://example.com",
        download_url="https://cdn.example.com/img.jpg",
        license_type="unsplash",
        license_url="https://unsplash.com/license",
        attribution="Test",
        photographer="Test",
        photographer_url="",
        width=400,
        height=300,
        local_path=path,
        file_size_bytes=path.stat().st_size,
        query="test",
    )


# ------------------------------------------------------------------ #
# QualityFilter
# ------------------------------------------------------------------ #


class TestQualityFilter:
    def test_accepts_good_image(self, tmp_path):
        record = _record_with_file(tmp_path, make_jpeg_bytes(400, 300))
        qf = QualityFilter(min_width=100, min_height=100, blur_threshold=None, min_file_size=0)
        passed, reason = qf.check(record)
        assert passed, reason

    def test_rejects_too_narrow(self, tmp_path):
        record = _record_with_file(tmp_path, make_jpeg_bytes(50, 300))
        qf = QualityFilter(min_width=200, min_height=100, min_file_size=0)
        passed, _ = qf.check(record)
        assert not passed

    def test_rejects_too_short(self, tmp_path):
        record = _record_with_file(tmp_path, make_jpeg_bytes(400, 50))
        qf = QualityFilter(min_width=100, min_height=200, min_file_size=0)
        passed, _ = qf.check(record)
        assert not passed

    def test_rejects_bad_aspect_ratio(self, tmp_path):
        # 1600x100 → aspect ratio 16 > 4
        record = _record_with_file(tmp_path, make_jpeg_bytes(1600, 100))
        qf = QualityFilter(min_width=50, min_height=50, max_aspect_ratio=4.0, min_file_size=0)
        passed, reason = qf.check(record)
        assert not passed
        assert "aspect ratio" in reason

    def test_rejects_missing_file(self, tmp_path):
        record = _record_with_file(tmp_path, make_jpeg_bytes())
        record.local_path = tmp_path / "nonexistent.jpg"
        qf = QualityFilter()
        passed, reason = qf.check(record)
        assert not passed

    def test_rejects_no_local_path(self):
        record = ImageRecord(
            source="unsplash", source_id="x", url="", download_url="",
            license_type="unsplash", license_url="", attribution="", photographer="",
            photographer_url="", width=800, height=600, query="",
        )
        qf = QualityFilter()
        passed, reason = qf.check(record)
        assert not passed

    def test_rejects_too_small_file(self, tmp_path):
        # 1-byte file
        path = tmp_path / "tiny.jpg"
        path.write_bytes(b"x")
        record = _record_with_file(tmp_path, make_jpeg_bytes())
        record.local_path = path
        qf = QualityFilter(min_file_size=1024)
        passed, reason = qf.check(record)
        assert not passed

    def test_rejects_blurry_image(self, tmp_path):
        record = _record_with_file(tmp_path, make_blurry_jpeg_bytes())
        qf = QualityFilter(min_width=50, min_height=50, blur_threshold=100.0, min_file_size=0)
        # Solid-colour image has very low Laplacian variance
        passed, reason = qf.check(record)
        assert not passed
        assert "blurry" in reason

    def test_accepts_sharp_image(self, tmp_path):
        record = _record_with_file(tmp_path, make_sharp_jpeg_bytes(400, 300))
        qf = QualityFilter(min_width=100, min_height=100, blur_threshold=10.0, min_file_size=0)
        passed, reason = qf.check(record)
        assert passed, reason

    def test_blur_check_skipped_when_threshold_none(self, tmp_path):
        record = _record_with_file(tmp_path, make_blurry_jpeg_bytes())
        qf = QualityFilter(min_width=50, min_height=50, blur_threshold=None, min_file_size=0)
        passed, _ = qf.check(record)
        assert passed

    def test_compute_quality_signals(self, tmp_path):
        record = _record_with_file(tmp_path, make_sharp_jpeg_bytes())
        qf = QualityFilter(blur_threshold=None)
        signals = qf.compute_quality_signals(record)
        assert "width" in signals
        assert "height" in signals
        assert "file_size_bytes" in signals

    def test_compute_blur_numpy_fallback(self, tmp_path):
        """_compute_blur falls back to numpy variance when _HAS_CV2 is False."""
        record = _record_with_file(tmp_path, make_sharp_jpeg_bytes())
        qf = QualityFilter(blur_threshold=None)

        import idc.filters.quality as quality_mod
        with patch.object(quality_mod, "_HAS_CV2", False):
            img = Image.open(record.local_path)
            score = qf._compute_blur(record.local_path, img)

        assert score is not None
        assert score > 0

    def test_compute_blur_numpy_exception_returns_none(self, tmp_path):
        """_compute_blur returns None when numpy raises inside the fallback path."""
        import numpy as np
        import idc.filters.quality as quality_mod

        record = _record_with_file(tmp_path, make_sharp_jpeg_bytes())
        qf = QualityFilter(blur_threshold=None)

        with patch.object(quality_mod, "_HAS_CV2", False):
            with patch.object(np, "array", side_effect=RuntimeError("numpy boom")):
                img = Image.open(record.local_path)
                score = qf._compute_blur(record.local_path, img)

        assert score is None


# ------------------------------------------------------------------ #
# CLIPFilter
# ------------------------------------------------------------------ #


def _clip_record(local_path=None, query="cat"):
    return ImageRecord(
        source="unsplash", source_id="x", url="", download_url="",
        license_type="unsplash", license_url="", attribution="", photographer="",
        photographer_url="", width=100, height=100, query=query,
        local_path=local_path,
    )


@contextmanager
def _mock_torch(similarity: float):
    """Patch sys.modules['torch'] with a fake that returns the given cosine similarity."""
    fake_torch = MagicMock()

    # torch.no_grad() context manager
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    fake_torch.no_grad.return_value = ctx

    # Image features
    img_feat = MagicMock()
    norm_img = MagicMock()
    img_feat.__truediv__ = MagicMock(return_value=norm_img)
    img_feat.norm.return_value = MagicMock()

    # Text features
    text_feat = MagicMock()
    norm_text = MagicMock()
    text_feat.__truediv__ = MagicMock(return_value=norm_text)
    text_feat.norm.return_value = MagicMock()

    # (img @ text.T).item() → similarity
    matmul_result = MagicMock()
    matmul_result.item.return_value = similarity
    norm_img.__matmul__ = MagicMock(return_value=matmul_result)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        yield fake_torch, img_feat, text_feat


def _wire_clip_instance(cf: CLIPFilter, img_feat, text_feat):
    """Set fake model internals on a CLIPFilter instance."""
    cf._device = "cpu"
    cf._model = MagicMock()
    cf._model.encode_image.return_value = img_feat
    cf._model.encode_text.return_value = text_feat
    cf._preprocess = MagicMock(return_value=MagicMock())
    cf._tokenizer = MagicMock(return_value=MagicMock())


class TestCLIPFilter:
    def test_no_local_path(self):
        cf = CLIPFilter()
        record = _clip_record(local_path=None)
        passed, reason = cf.check(record)
        assert not passed
        assert reason == "no local file"

    def test_file_not_found(self, tmp_path):
        cf = CLIPFilter()
        record = _clip_record(local_path=tmp_path / "ghost.jpg")
        passed, reason = cf.check(record)
        assert not passed
        assert reason == "file not found"

    def test_import_error_fails_open(self, tmp_path):
        """When CLIP is not installed, check() returns True (fail-open)."""
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        cf = CLIPFilter()
        record = _clip_record(local_path=path)
        with patch.object(cf, "_load_model", side_effect=ImportError("no clip")):
            passed, reason = cf.check(record)
        assert passed
        assert reason == ""

    def test_below_threshold(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        cf = CLIPFilter(threshold=0.3)
        record = _clip_record(local_path=path)

        with _mock_torch(similarity=0.1) as (fake_torch, img_feat, text_feat):
            _wire_clip_instance(cf, img_feat, text_feat)
            with patch.object(cf, "_load_model"):  # skip real load
                passed, reason = cf.check(record)

        assert not passed
        assert "0.1" in reason or "CLIP" in reason

    def test_above_threshold(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        cf = CLIPFilter(threshold=0.2)
        record = _clip_record(local_path=path)

        with _mock_torch(similarity=0.5) as (fake_torch, img_feat, text_feat):
            _wire_clip_instance(cf, img_feat, text_feat)
            with patch.object(cf, "_load_model"):
                passed, reason = cf.check(record)

        assert passed
        assert reason == ""

    def test_decode_error_fails_open(self, tmp_path):
        """Errors during image inference are swallowed (fail-open)."""
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        cf = CLIPFilter()
        record = _clip_record(local_path=path)

        with _mock_torch(similarity=0.5) as (fake_torch, img_feat, text_feat):
            _wire_clip_instance(cf, img_feat, text_feat)
            cf._preprocess.side_effect = RuntimeError("decode boom")
            with patch.object(cf, "_load_model"):
                passed, reason = cf.check(record)

        assert passed
        assert reason == ""

    def test_text_features_cached(self, tmp_path):
        """encode_text is only called once for repeated queries."""
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        cf = CLIPFilter(threshold=0.1)
        record = _clip_record(local_path=path, query="cat")

        with _mock_torch(similarity=0.5) as (fake_torch, img_feat, text_feat):
            _wire_clip_instance(cf, img_feat, text_feat)
            with patch.object(cf, "_load_model"):
                cf.check(record)
                cf.check(record)  # second call with same query

        # encode_text is inside _get_text_features which caches by query
        assert cf._model.encode_text.call_count == 1


# ------------------------------------------------------------------ #
# Deduplicator
# ------------------------------------------------------------------ #


class TestDeduplicator:
    def test_first_image_is_unique(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes(400, 300, color=(10, 20, 30)))

        record = _record_with_file(tmp_path, make_jpeg_bytes(400, 300, color=(10, 20, 30)))
        dedup = Deduplicator(threshold=10)
        is_unique, _ = dedup.check_and_add(record)
        assert is_unique

    def test_identical_image_is_duplicate(self, tmp_path):
        img_bytes = make_jpeg_bytes(400, 300, color=(55, 66, 77))

        record1 = _record_with_file(tmp_path, img_bytes, source_id="img1")
        record2 = _record_with_file(tmp_path, img_bytes, source_id="img2")

        dedup = Deduplicator(threshold=10)
        is_unique1, _ = dedup.check_and_add(record1)
        is_unique2, reason = dedup.check_and_add(record2)

        assert is_unique1
        assert not is_unique2
        assert "duplicate" in reason

    def test_different_images_are_unique(self, tmp_path):
        img1 = _record_with_file(tmp_path, make_sharp_jpeg_bytes(400, 300), source_id="sharp")
        img2 = _record_with_file(tmp_path, make_jpeg_bytes(400, 300, color=(200, 10, 10)), source_id="red")

        dedup = Deduplicator(threshold=5)
        assert dedup.check_and_add(img1)[0]
        assert dedup.check_and_add(img2)[0]

    def test_load_existing_seeds_seen_hashes(self, tmp_path):
        import imagehash
        from PIL import Image

        img_bytes = make_jpeg_bytes(400, 300, color=(99, 88, 77))
        record = _record_with_file(tmp_path, img_bytes, source_id="dup")

        # Compute the hash externally
        img = Image.open(tmp_path / "dup.jpg")
        phash = str(imagehash.phash(img))

        dedup = Deduplicator(threshold=10)
        dedup.load_existing([phash])

        # Now checking the record should see it as duplicate
        is_unique, _ = dedup.check_and_add(record)
        assert not is_unique

    def test_compute_hash_returns_string(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(make_jpeg_bytes())
        dedup = Deduplicator()
        h = dedup.compute_hash(path)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_compute_hash_nonexistent_file(self, tmp_path):
        dedup = Deduplicator()
        h = dedup.compute_hash(tmp_path / "no_such_file.jpg")
        assert h is None

    def test_allows_through_when_no_hash(self):
        record = ImageRecord(
            source="unsplash", source_id="x", url="", download_url="",
            license_type="unsplash", license_url="", attribution="", photographer="",
            photographer_url="", width=800, height=600, query="",
        )
        # No local_path → no hash → should be allowed through
        dedup = Deduplicator()
        is_unique, _ = dedup.check_and_add(record)
        assert is_unique
