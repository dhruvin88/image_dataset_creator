from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .exporters.base import BaseExporter
from .filters.dedup import Deduplicator
from .filters.quality import QualityFilter
from .manifest import Manifest
from .models import ImageRecord
from .sources.base import ImageSource

console = Console()


@dataclass
class DownloadSummary:
    """Tracks outcomes for a download() call."""
    accepted: int = 0
    skipped_existing: int = 0
    failed_download: int = 0
    failed_quality: int = 0
    failed_dedup: int = 0
    failures: List[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.accepted + self.skipped_existing
            + self.failed_download + self.failed_quality + self.failed_dedup
        )

    def print(self) -> None:
        table = Table(title="Download Summary", show_header=True)
        table.add_column("Outcome")
        table.add_column("Count", justify="right")
        table.add_row("[green]Accepted[/green]", str(self.accepted))
        if self.skipped_existing:
            table.add_row("[dim]Skipped (already downloaded)[/dim]", str(self.skipped_existing))
        if self.failed_download:
            table.add_row("[red]Failed (download error)[/red]", str(self.failed_download))
        if self.failed_quality:
            table.add_row("[yellow]Rejected (quality)[/yellow]", str(self.failed_quality))
        if self.failed_dedup:
            table.add_row("[yellow]Rejected (duplicate)[/yellow]", str(self.failed_dedup))
        console.print(table)

    def save_log(self, path: Path) -> None:
        """Save per-failure details to a JSONL file."""
        if not self.failures:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.failures:
                f.write(json.dumps(entry) + "\n")


class DatasetBuilder:
    """
    Orchestrates the full pipeline:
        add_source → search → download → quality filter → dedup → export

    Downloads are executed concurrently via asyncio (up to max_workers at once).
    By default, download() skips images already present in the manifest
    (resume-safe across interrupted runs).
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        dataset_name: str = "dataset",
        max_workers: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.max_workers = max_workers

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(self.output_dir / "manifest.db")

        self._sources: List[ImageSource] = []
        self._quality_filter: Optional[QualityFilter] = None
        self._deduplicator: Optional[Deduplicator] = None
        self._clip_filter = None
        self.last_summary: Optional[DownloadSummary] = None

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def add_source(self, source: ImageSource) -> DatasetBuilder:
        self._sources.append(source)
        return self

    def add_filter(self, f) -> DatasetBuilder:
        if isinstance(f, QualityFilter):
            self._quality_filter = f
        elif isinstance(f, Deduplicator):
            self._deduplicator = f
        else:
            # CLIPFilter or any future post-download filter
            self._clip_filter = f
        return self

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        """Search all configured sources. Distributes count evenly across sources."""
        if not self._sources:
            raise ValueError("No sources configured. Call add_source() first.")

        n = len(self._sources)
        per_source = count // n
        remainder = count - per_source * n
        all_records: List[ImageRecord] = []

        with tqdm(total=n, desc=f"Searching '{query}'", unit="source", leave=False) as pbar:
            for i, source in enumerate(self._sources):
                target = per_source + (1 if i < remainder else 0)
                pbar.set_description(f"Searching {source.name} for '{query}'")

                # Per-page progress bar for this source
                with tqdm(
                    total=target,
                    desc=f"  {source.name}",
                    unit="img",
                    leave=False,
                ) as img_pbar:
                    try:
                        records = source.search(
                            query, target, on_page=img_pbar.update, **kwargs
                        )
                        for r in records:
                            r.dataset_name = self.dataset_name
                        all_records.extend(records)
                    except Exception as exc:
                        console.print(f"[yellow]Warning: {source.name} search failed: {exc}[/yellow]")

                pbar.update(1)

        return all_records

    # ------------------------------------------------------------------ #
    # Download + filter + dedup  (async core, sync public API)
    # ------------------------------------------------------------------ #

    def download(
        self,
        records: List[ImageRecord],
        label: Optional[str] = None,
        skip_existing: bool = True,
        save_failure_log: bool = False,
    ) -> List[ImageRecord]:
        """
        Download records, apply quality filter and deduplication.

        Internally uses asyncio for concurrent downloads (up to max_workers).

        skip_existing: if True (default), records already in the manifest are
        skipped — making repeated runs resume from where they left off.
        save_failure_log: if True, writes download_failures.jsonl to output_dir.

        Returns the list of accepted ImageRecords (including previously cached ones).
        """
        return asyncio.run(self._adownload(records, label, skip_existing, save_failure_log))

    async def _adownload(
        self,
        records: List[ImageRecord],
        label: Optional[str] = None,
        skip_existing: bool = True,
        save_failure_log: bool = False,
    ) -> List[ImageRecord]:
        images_dir = self.output_dir / "images"
        if label:
            images_dir = images_dir / label
        images_dir.mkdir(parents=True, exist_ok=True)

        if self._deduplicator:
            self._deduplicator.load_existing(self.manifest.get_phashes())

        accepted: List[ImageRecord] = []
        summary = DownloadSummary()
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.max_workers)

        def _source_for(record: ImageRecord) -> Optional[ImageSource]:
            for s in self._sources:
                if s.name == record.source:
                    return s
            return self._sources[0] if self._sources else None

        async def process(record: ImageRecord) -> Optional[ImageRecord]:
            async with sem:
                # --- Resume: skip if already in manifest with file on disk ---
                if skip_existing and self.manifest.has_source_id(record.source, record.source_id):
                    existing = self.manifest.get_by_source_id(record.source, record.source_id)
                    if existing and existing.local_path and Path(existing.local_path).exists():
                        async with lock:
                            accepted.append(existing)
                            summary.skipped_existing += 1
                        return existing

                source = _source_for(record)
                if source is None:
                    async with lock:
                        summary.failed_download += 1
                        summary.failures.append({
                            "source_id": record.source_id, "source": record.source,
                            "reason": "no source adapter found",
                        })
                    return None

                # --- Download ---
                try:
                    local_path = await source.adownload(record, images_dir)
                except Exception as exc:
                    console.print(f"[dim]Download failed for {record.source_id}: {exc}[/dim]")
                    async with lock:
                        summary.failed_download += 1
                        summary.failures.append({
                            "source_id": record.source_id, "source": record.source,
                            "reason": f"download error: {exc}",
                        })
                    return None

                record.local_path = local_path
                record.file_size_bytes = local_path.stat().st_size

                # --- Quality signals (CPU-bound: run in thread) ---
                if self._quality_filter:
                    try:
                        img = await asyncio.to_thread(Image.open, local_path)
                        signals = await asyncio.to_thread(
                            self._quality_filter.compute_quality_signals, record, img
                        )
                        record.blur_score = signals.get("blur_score", record.blur_score)
                        record.format = signals.get("format", record.format)
                        record.file_size_bytes = signals.get("file_size_bytes", record.file_size_bytes)
                    except Exception:
                        pass

                    passed, reason = await asyncio.to_thread(self._quality_filter.check, record)
                    if not passed:
                        local_path.unlink(missing_ok=True)
                        async with lock:
                            summary.failed_quality += 1
                            summary.failures.append({
                                "source_id": record.source_id, "source": record.source,
                                "reason": f"quality: {reason}",
                            })
                        return None

                # --- CLIP semantic filter (CPU/GPU-bound: run in thread) ---
                if self._clip_filter:
                    passed, reason = await asyncio.to_thread(self._clip_filter.check, record)
                    if not passed:
                        local_path.unlink(missing_ok=True)
                        async with lock:
                            summary.failed_quality += 1
                            summary.failures.append({
                                "source_id": record.source_id, "source": record.source,
                                "reason": f"clip: {reason}",
                            })
                        return None

                # --- Dedup (check+add must be atomic under lock) ---
                if self._deduplicator:
                    phash = await asyncio.to_thread(self._deduplicator.compute_hash, local_path)
                    record.phash = phash
                    async with lock:
                        is_unique, _ = self._deduplicator.check_and_add(record)
                    if not is_unique:
                        local_path.unlink(missing_ok=True)
                        async with lock:
                            summary.failed_dedup += 1
                        return None

                async with lock:
                    self.manifest.add(record)
                    accepted.append(record)
                    summary.accepted += 1

                return record

        with tqdm(total=len(records), desc="Downloading", unit="img") as pbar:
            async def process_with_progress(record: ImageRecord) -> Optional[ImageRecord]:
                result = await process(record)
                pbar.update(1)
                return result

            await asyncio.gather(*[process_with_progress(r) for r in records])

        summary.print()
        if save_failure_log and summary.failures:
            log_path = self.output_dir / "download_failures.jsonl"
            summary.save_log(log_path)
            console.print(f"[dim]Failure log saved to {log_path}[/dim]")

        self.last_summary = summary
        return accepted

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export(self, exporter: BaseExporter, output_dir: Optional[Path] = None) -> None:
        records = self.manifest.get_all()
        out = output_dir or self.output_dir
        exporter.export(records, out)
        console.print(f"[green]Exported {len(records)} images to {out}[/green]")
