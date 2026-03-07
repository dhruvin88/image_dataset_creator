"""Tests for dataset exporters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from idc.exporters.coco import COCOExporter
from idc.exporters.raw import RawExporter
from idc.exporters.yolo import YOLOExporter
from idc.models import ImageRecord

from .conftest import make_jpeg_bytes


def _make_records_with_files(tmp_path: Path, n: int = 3) -> List[ImageRecord]:
    records = []
    for i in range(n):
        path = tmp_path / f"img_{i:03d}.jpg"
        path.write_bytes(make_jpeg_bytes(400, 300))
        records.append(
            ImageRecord(
                source="unsplash",
                source_id=f"id{i}",
                url=f"https://unsplash.com/photos/id{i}",
                download_url=f"https://cdn.example.com/{i}.jpg",
                license_type="unsplash",
                license_url="https://unsplash.com/license",
                attribution=f"Photo by User{i} on Unsplash",
                photographer=f"User{i}",
                photographer_url=f"https://unsplash.com/@user{i}",
                width=400,
                height=300,
                local_path=path,
                file_size_bytes=path.stat().st_size,
                query="dogs",
            )
        )
    return records


# ------------------------------------------------------------------ #
# RawExporter
# ------------------------------------------------------------------ #


class TestRawExporter:
    def test_creates_images_dir_and_metadata(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src)

        RawExporter().export(records, out)

        assert (out / "images").is_dir()
        assert (out / "metadata.jsonl").exists()

    def test_metadata_has_correct_count(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=4)

        RawExporter().export(records, out)

        lines = (out / "metadata.jsonl").read_text().strip().split("\n")
        assert len(lines) == 4

    def test_metadata_is_valid_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=2)

        RawExporter().export(records, out)

        for line in (out / "metadata.jsonl").read_text().strip().split("\n"):
            data = json.loads(line)
            assert "id" in data
            assert "attribution" in data
            assert "license_type" in data

    def test_images_are_copied(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=2)

        RawExporter().export(records, out)

        images = list((out / "images").glob("*.jpg"))
        assert len(images) == 2

    def test_skips_records_without_local_path(self, tmp_path):
        out = tmp_path / "output"
        record = ImageRecord(
            source="unsplash", source_id="nolocalpath",
            url="", download_url="", license_type="unsplash", license_url="",
            attribution="", photographer="", photographer_url="", width=100, height=100, query="",
        )
        RawExporter().export([record], out)
        assert not (out / "metadata.jsonl").exists() or (out / "metadata.jsonl").read_text() == ""


# ------------------------------------------------------------------ #
# YOLOExporter
# ------------------------------------------------------------------ #


class TestYOLOExporter:
    def test_creates_directory_structure(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=5)

        YOLOExporter().export(records, out)

        assert (out / "images" / "train").is_dir()
        assert (out / "images" / "val").is_dir()
        assert (out / "labels" / "train").is_dir()
        assert (out / "labels" / "val").is_dir()
        assert (out / "dataset.yaml").exists()

    def test_dataset_yaml_content(self, tmp_path):
        import yaml

        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=5)

        YOLOExporter(class_names=["dog", "cat"]).export(records, out)

        with open(out / "dataset.yaml") as f:
            cfg = yaml.safe_load(f)

        assert cfg["nc"] == 2
        assert cfg["names"] == ["dog", "cat"]
        assert "train" in cfg
        assert "val" in cfg

    def test_labels_are_empty_files(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=3)

        YOLOExporter().export(records, out)

        for label_file in (out / "labels" / "train").glob("*.txt"):
            assert label_file.read_text() == ""

    def test_val_split(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=10)

        YOLOExporter(val_split=0.2).export(records, out)

        n_val = len(list((out / "images" / "val").glob("*.jpg")))
        n_train = len(list((out / "images" / "train").glob("*.jpg")))
        assert n_val >= 1
        assert n_val + n_train == 10

    def test_test_split_creates_test_dir(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=10)

        YOLOExporter(val_split=0.2, test_split=0.1).export(records, out)

        assert (out / "images" / "test").is_dir()
        assert (out / "labels" / "test").is_dir()
        n_test = len(list((out / "images" / "test").glob("*.jpg")))
        assert n_test >= 1

    def test_test_split_in_dataset_yaml(self, tmp_path):
        import yaml
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=10)
        YOLOExporter(val_split=0.2, test_split=0.1).export(records, out)
        with open(out / "dataset.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "test" in cfg


# ------------------------------------------------------------------ #
# RawExporter with splits
# ------------------------------------------------------------------ #


class TestRawExporterSplits:
    def test_split_creates_subdirs(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=10)

        RawExporter(val_split=0.2, test_split=0.1).export(records, out)

        assert (out / "images" / "train").is_dir()
        assert (out / "images" / "val").is_dir()
        assert (out / "images" / "test").is_dir()

    def test_split_adds_split_field_to_metadata(self, tmp_path):
        import json
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=5)

        RawExporter(val_split=0.2).export(records, out)

        lines = (out / "metadata.jsonl").read_text().strip().split("\n")
        splits = {json.loads(l)["split"] for l in lines}
        assert "train" in splits
        assert "val" in splits

    def test_no_split_produces_flat_structure(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=3)

        RawExporter(val_split=0.0).export(records, out)

        assert (out / "images").is_dir()
        assert not (out / "images" / "train").exists()


# ------------------------------------------------------------------ #
# COCOExporter with splits
# ------------------------------------------------------------------ #


class TestCOCOExporterSplits:
    def test_split_creates_separate_annotation_files(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=10)

        COCOExporter(val_split=0.2).export(records, out)

        assert (out / "annotations" / "train.json").exists()
        assert (out / "annotations" / "val.json").exists()

    def test_no_split_uses_instances_json(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=3)

        COCOExporter(val_split=0.0).export(records, out)

        assert (out / "annotations" / "instances.json").exists()
        assert not (out / "annotations" / "train.json").exists()


# ------------------------------------------------------------------ #
# CSVExporter
# ------------------------------------------------------------------ #


class TestCSVExporter:
    def test_creates_csv_file(self, tmp_path):
        from idc.exporters.csv_exporter import CSVExporter
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=3)

        CSVExporter().export(records, out)

        assert (out / "dataset.csv").exists()

    def test_csv_has_header_and_rows(self, tmp_path):
        import csv as csv_module
        from idc.exporters.csv_exporter import CSVExporter
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=4)

        CSVExporter().export(records, out)

        with open(out / "dataset.csv", newline="") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4
        assert "id" in rows[0]
        assert "attribution" in rows[0]
        assert "license_type" in rows[0]
        assert "source" in rows[0]

    def test_empty_records_produces_no_file(self, tmp_path):
        from idc.exporters.csv_exporter import CSVExporter
        out = tmp_path / "output"
        CSVExporter().export([], out)
        assert not (out / "dataset.csv").exists()

    def test_custom_filename(self, tmp_path):
        from idc.exporters.csv_exporter import CSVExporter
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=1)

        CSVExporter(filename="my_data.csv").export(records, out)

        assert (out / "my_data.csv").exists()


# ------------------------------------------------------------------ #
# COCOExporter
# ------------------------------------------------------------------ #


class TestCOCOExporter:
    def test_creates_structure(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=3)

        COCOExporter().export(records, out)

        assert (out / "images").is_dir()
        assert (out / "annotations" / "instances.json").exists()

    def test_instances_json_has_correct_image_count(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=4)

        COCOExporter().export(records, out)

        with open(out / "annotations" / "instances.json") as f:
            data = json.load(f)

        assert len(data["images"]) == 4

    def test_instances_json_has_required_coco_keys(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=2)

        COCOExporter(dataset_name="test_ds").export(records, out)

        with open(out / "annotations" / "instances.json") as f:
            data = json.load(f)

        assert "info" in data
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert data["info"]["description"] == "test_ds"

    def test_image_entries_have_attribution(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "output"
        records = _make_records_with_files(src, n=1)

        COCOExporter().export(records, out)

        with open(out / "annotations" / "instances.json") as f:
            data = json.load(f)

        img = data["images"][0]
        assert "attribution" in img
        assert "license" in img
        assert img["source"] == "unsplash"


# ------------------------------------------------------------------ #
# HuggingFaceExporter (mocked datasets library)
# ------------------------------------------------------------------ #


def _make_mock_hf_datasets():
    """Build a mock that mimics the `datasets` library interface."""
    mock_ds = MagicMock()

    # Features / Value / Sequence
    mock_ds.Value = MagicMock(return_value=MagicMock())
    mock_ds.Sequence = MagicMock(return_value=MagicMock())
    mock_ds.Features = MagicMock(return_value=MagicMock())

    # Dataset.from_dict returns a mock dataset
    fake_dataset = MagicMock()
    fake_dataset.save_to_disk = MagicMock()
    mock_ds.Dataset = MagicMock()
    mock_ds.Dataset.from_dict = MagicMock(return_value=fake_dataset)

    # DatasetDict wraps splits
    fake_dict = MagicMock()
    fake_dict.save_to_disk = MagicMock()
    mock_ds.DatasetDict = MagicMock(return_value=fake_dict)

    return mock_ds, fake_dataset, fake_dict


class TestHuggingFaceExporter:
    def test_no_split_calls_dataset_from_dict(self, tmp_path):
        from idc.exporters.huggingface import HuggingFaceExporter

        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "hf_out"
        records = _make_records_with_files(src, n=3)

        mock_ds, fake_dataset, _ = _make_mock_hf_datasets()

        with patch.dict("sys.modules", {"datasets": mock_ds}):
            HuggingFaceExporter(val_split=0.0).export(records, out)

        mock_ds.Dataset.from_dict.assert_called_once()
        fake_dataset.save_to_disk.assert_called_once_with(str(out))

    def test_with_split_calls_dataset_dict(self, tmp_path):
        from idc.exporters.huggingface import HuggingFaceExporter

        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "hf_split_out"
        records = _make_records_with_files(src, n=5)

        mock_ds, _, fake_dict = _make_mock_hf_datasets()

        with patch.dict("sys.modules", {"datasets": mock_ds}):
            HuggingFaceExporter(val_split=0.2).export(records, out)

        mock_ds.DatasetDict.assert_called_once()
        fake_dict.save_to_disk.assert_called_once_with(str(out))

    def test_with_test_split_includes_test_key(self, tmp_path):
        from idc.exporters.huggingface import HuggingFaceExporter

        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "hf_test_out"
        records = _make_records_with_files(src, n=10)

        mock_ds, _, fake_dict = _make_mock_hf_datasets()

        with patch.dict("sys.modules", {"datasets": mock_ds}):
            HuggingFaceExporter(val_split=0.2, test_split=0.1).export(records, out)

        # DatasetDict was called with a dict containing "test" key
        call_kwargs = mock_ds.DatasetDict.call_args
        split_dict = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]
        assert "test" in split_dict
        assert "train" in split_dict
        assert "validation" in split_dict

    def test_missing_datasets_library_raises_import_error(self, tmp_path):
        from idc.exporters.huggingface import HuggingFaceExporter

        out = tmp_path / "hf_missing"
        records = []

        import sys
        with patch.dict("sys.modules", {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                HuggingFaceExporter().export(records, out)
