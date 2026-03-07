"""Shared test fixtures."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from idc.models import ImageRecord


# ------------------------------------------------------------------ #
# Image helpers
# ------------------------------------------------------------------ #


def make_jpeg_bytes(width: int = 400, height: int = 300, color: tuple = (100, 150, 200)) -> bytes:
    """Create a minimal JPEG image as bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_sharp_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    """Create a high-contrast (sharp) JPEG image."""
    import numpy as np

    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_blurry_jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    """Create a solid-colour (blurry) JPEG — Laplacian variance ≈ 0."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_record() -> ImageRecord:
    return ImageRecord(
        source="unsplash",
        source_id="abc123",
        url="https://unsplash.com/photos/abc123",
        download_url="https://images.unsplash.com/photo-abc123",
        license_type="unsplash",
        license_url="https://unsplash.com/license",
        attribution="Photo by Jane Doe on Unsplash",
        photographer="Jane Doe",
        photographer_url="https://unsplash.com/@janedoe",
        width=800,
        height=600,
        tags=["dog", "golden retriever"],
        description="A golden retriever in the park",
        query="golden retriever",
    )


@pytest.fixture
def sample_record_with_file(tmp_path: Path, sample_record: ImageRecord) -> ImageRecord:
    """A sample record that has a local file on disk."""
    img_path = tmp_path / "unsplash_abc123.jpg"
    img_path.write_bytes(make_jpeg_bytes())
    record = sample_record.model_copy()
    record.local_path = img_path
    record.file_size_bytes = img_path.stat().st_size
    return record


@pytest.fixture
def sharp_image_file(tmp_path: Path) -> Path:
    path = tmp_path / "sharp.jpg"
    path.write_bytes(make_sharp_jpeg_bytes())
    return path


@pytest.fixture
def blurry_image_file(tmp_path: Path) -> Path:
    path = tmp_path / "blurry.jpg"
    path.write_bytes(make_blurry_jpeg_bytes())
    return path
