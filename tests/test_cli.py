"""Tests for the CLI using Click's test runner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from idc.cli import cli
from idc.models import ImageRecord

from .conftest import make_jpeg_bytes


def _make_record(tmp_path: Path, source_id: str = "test001") -> ImageRecord:
    path = tmp_path / f"{source_id}.jpg"
    path.write_bytes(make_jpeg_bytes())
    return ImageRecord(
        source="unsplash",
        source_id=source_id,
        url="https://unsplash.com/photos/test001",
        download_url="https://images.unsplash.com/test001",
        license_type="unsplash",
        license_url="https://unsplash.com/license",
        attribution="Photo by CLI Test on Unsplash",
        photographer="CLI Test",
        photographer_url="",
        width=400,
        height=300,
        local_path=path,
        file_size_bytes=path.stat().st_size,
        query="golden retriever",
    )


class TestConfigCommands:
    def test_config_set_prints_saved(self, monkeypatch):
        runner = CliRunner()
        with patch("idc.config.set_api_key") as mock_set:
            result = runner.invoke(cli, ["config", "set", "--unsplash-key", "mykey123"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower()
        mock_set.assert_called_once_with("unsplash", "mykey123")

    def test_config_set_no_keys_warns(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set"])
        assert result.exit_code == 0
        assert "no keys" in result.output.lower()

    def test_config_show_displays_sources(self):
        runner = CliRunner()
        with patch("idc.config.get_all_keys", return_value={"unsplash": "key", "pexels": None, "pixabay": None}):
            result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "unsplash" in result.output.lower()


class TestInfoCommand:
    def test_info_empty_dataset(self, tmp_path):
        from idc.manifest import Manifest

        Manifest(tmp_path / "manifest.db")  # Create empty manifest
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(tmp_path)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_info_with_records(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        manifest.add(_make_record(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(tmp_path)])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_info_missing_manifest_aborts(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(tmp_path)])
        assert result.exit_code != 0


class TestReportCommand:
    def test_report_creates_csv(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        manifest.add(_make_record(tmp_path))

        runner = CliRunner()
        output_csv = tmp_path / "report.csv"
        result = runner.invoke(cli, ["report", str(tmp_path), "--output", str(output_csv)])

        assert result.exit_code == 0
        assert output_csv.exists()
        content = output_csv.read_text()
        assert "attribution" in content
        assert "CLI Test" in content

    def test_report_empty_dataset(self, tmp_path):
        from idc.manifest import Manifest

        Manifest(tmp_path / "manifest.db")
        runner = CliRunner()
        output_csv = tmp_path / "report.csv"
        result = runner.invoke(cli, ["report", str(tmp_path), "--output", str(output_csv)])
        assert result.exit_code == 0
        assert output_csv.exists()


class TestExportCommand:
    def test_export_raw_format(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        manifest.add(_make_record(tmp_path))

        runner = CliRunner()
        out_dir = tmp_path / "exported"
        result = runner.invoke(cli, ["export", str(tmp_path), "--format", "raw", "--output", str(out_dir)])

        assert result.exit_code == 0
        assert out_dir.is_dir()

    def test_export_yolo_format(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        for i in range(5):
            manifest.add(_make_record(tmp_path, source_id=f"id{i}"))

        runner = CliRunner()
        out_dir = tmp_path / "yolo_export"
        result = runner.invoke(cli, ["export", str(tmp_path), "--format", "yolo", "--output", str(out_dir)])

        assert result.exit_code == 0
        assert (out_dir / "dataset.yaml").exists()

    def test_export_coco_format(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        manifest.add(_make_record(tmp_path))

        runner = CliRunner()
        out_dir = tmp_path / "coco_export"
        result = runner.invoke(
            cli,
            ["export", str(tmp_path), "--format", "coco", "--val-split", "0", "--output", str(out_dir)],
        )

        assert result.exit_code == 0
        assert (out_dir / "annotations" / "instances.json").exists()


class TestFilterCommand:
    def test_filter_removes_records_from_manifest(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        # Add a record pointing to a very small file that won't pass min-file-size
        record = _make_record(tmp_path)
        img_path = tmp_path / f"{record.source_id}.jpg"
        img_path.write_bytes(b"x" * 5)  # 5 bytes, will fail file size check
        record.local_path = img_path
        manifest.add(record)
        assert manifest.count() == 1

        runner = CliRunner()
        result = runner.invoke(cli, ["filter", str(tmp_path), "--min-width", "100"])
        assert result.exit_code == 0
        assert "removed" in result.output.lower()
        assert manifest.count() == 0  # record was actually removed

    def test_filter_dry_run_does_not_modify_manifest(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        record = _make_record(tmp_path)
        img_path = tmp_path / f"{record.source_id}.jpg"
        img_path.write_bytes(b"x" * 5)
        record.local_path = img_path
        manifest.add(record)

        runner = CliRunner()
        result = runner.invoke(cli, ["filter", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
        assert manifest.count() == 1  # unchanged

    def test_filter_keep_files_does_not_delete_image(self, tmp_path):
        from idc.manifest import Manifest

        manifest = Manifest(tmp_path / "manifest.db")
        record = _make_record(tmp_path)
        img_path = tmp_path / f"{record.source_id}.jpg"
        img_path.write_bytes(b"x" * 5)
        record.local_path = img_path
        manifest.add(record)

        runner = CliRunner()
        runner.invoke(cli, ["filter", str(tmp_path), "--keep-files"])
        assert img_path.exists()  # file kept even though record removed

    def test_filter_deletes_files_by_default(self, tmp_path):
        from idc.manifest import Manifest
        from tests.conftest import make_jpeg_bytes

        manifest = Manifest(tmp_path / "manifest.db")
        record = _make_record(tmp_path)
        img_path = tmp_path / f"{record.source_id}.jpg"
        img_path.write_bytes(b"x" * 5)  # too small for default filter
        record.local_path = img_path
        manifest.add(record)

        runner = CliRunner()
        runner.invoke(cli, ["filter", str(tmp_path)])
        assert not img_path.exists()  # file deleted


class TestMergeCommand:
    def test_merge_combines_two_datasets(self, tmp_path):
        from idc.manifest import Manifest

        ds_a = tmp_path / "ds_a"
        ds_b = tmp_path / "ds_b"
        ds_a.mkdir()
        ds_b.mkdir()
        merged = tmp_path / "merged"

        manifest_a = Manifest(ds_a / "manifest.db")
        manifest_b = Manifest(ds_b / "manifest.db")
        manifest_a.add(_make_record(ds_a, source_id="a001"))
        manifest_b.add(_make_record(ds_b, source_id="b001"))

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(ds_a), str(ds_b), "--no-dedup", "--output", str(merged)])

        assert result.exit_code == 0
        assert "merge complete" in result.output.lower()
        from idc.manifest import Manifest as M
        out_manifest = M(merged / "manifest.db")
        assert out_manifest.count() == 2

    def test_merge_deduplicates_identical_images(self, tmp_path):
        from idc.manifest import Manifest

        ds_a = tmp_path / "ds_a"
        ds_b = tmp_path / "ds_b"
        ds_a.mkdir()
        ds_b.mkdir()
        merged = tmp_path / "merged"

        # Both datasets have the same image content — same phash
        record_a = _make_record(ds_a, source_id="dup001")
        record_b = _make_record(ds_b, source_id="dup001")

        # Give them phashes so dedup can compare
        import imagehash
        from PIL import Image
        img = Image.open(record_a.local_path)
        phash = str(imagehash.phash(img))
        record_a.phash = phash
        record_b.phash = phash

        Manifest(ds_a / "manifest.db").add(record_a)
        Manifest(ds_b / "manifest.db").add(record_b)

        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(ds_a), str(ds_b), "--dedup", "--output", str(merged)])

        assert result.exit_code == 0
        from idc.manifest import Manifest as M
        # One should be deduplicated
        assert M(merged / "manifest.db").count() == 1


class TestSearchCommand:
    def test_search_aborts_with_no_api_keys(self, tmp_path, monkeypatch):
        """Should abort cleanly when no sources can be configured."""
        monkeypatch.delenv("IDC_UNSPLASH_KEY", raising=False)
        monkeypatch.delenv("IDC_PEXELS_KEY", raising=False)

        with patch("idc.config.get_api_key", return_value=None):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["search", "dogs", "--sources", "unsplash,pexels", "--output", str(tmp_path / "ds")],
            )
        # Should abort because no valid sources
        assert result.exit_code != 0

    def test_search_with_mocked_sources(self, tmp_path):
        """End-to-end search with mocked network calls."""
        from idc.manifest import Manifest
        from idc.models import ImageRecord

        img_bytes = make_jpeg_bytes(400, 300)
        img_path = tmp_path / "downloaded.jpg"
        img_path.write_bytes(img_bytes)

        fake_record = ImageRecord(
            source="unsplash", source_id="fake001",
            url="https://unsplash.com/photos/fake001",
            download_url="https://images.unsplash.com/fake001",
            license_type="unsplash", license_url="https://unsplash.com/license",
            attribution="Photo by Fake on Unsplash", photographer="Fake",
            photographer_url="", width=400, height=300, query="dogs",
            local_path=img_path, file_size_bytes=img_path.stat().st_size,
        )

        with patch("idc.builder.DatasetBuilder.search", return_value=[fake_record]), \
             patch("idc.builder.DatasetBuilder.download", return_value=[fake_record]), \
             patch("idc.builder.DatasetBuilder.export"):
            with patch("idc.config.get_api_key", return_value="fake_key"):
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    ["search", "dogs", "--sources", "unsplash", "--output", str(tmp_path / "ds"), "--count", "1"],
                )

        assert result.exit_code == 0
