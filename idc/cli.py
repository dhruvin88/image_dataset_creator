from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()

FORMATS = ["raw", "yolo", "coco", "huggingface", "csv"]
SOURCES = ["unsplash", "pexels", "pixabay", "wikimedia", "openimages"]


@click.group()
@click.version_option(version="1.0.0", prog_name="idc")
def cli() -> None:
    """Image Dataset Creator — from query to training-ready dataset in one command."""


# ------------------------------------------------------------------ #
# config
# ------------------------------------------------------------------ #


@cli.group()
def config() -> None:
    """Manage API keys and configuration."""


@config.command("set")
@click.option("--unsplash-key", default=None, help="Unsplash API key")
@click.option("--pexels-key", default=None, help="Pexels API key")
@click.option("--pixabay-key", default=None, help="Pixabay API key")
def config_set(
    unsplash_key: Optional[str],
    pexels_key: Optional[str],
    pixabay_key: Optional[str],
) -> None:
    """Set API keys for image sources."""
    from .config import set_api_key

    pairs = [("unsplash", unsplash_key), ("pexels", pexels_key), ("pixabay", pixabay_key)]
    saved = False
    for source, key in pairs:
        if key:
            set_api_key(source, key)
            console.print(f"[green]{source.capitalize()} key saved.[/green]")
            saved = True

    if not saved:
        console.print("[yellow]No keys provided. Use --unsplash-key, --pexels-key, or --pixabay-key.[/yellow]")


@config.command("show")
def config_show() -> None:
    """Show configured API keys (masked)."""
    from .config import get_all_keys

    keys = get_all_keys()
    table = Table(title="API Keys")
    table.add_column("Source")
    table.add_column("Status")
    for source, key in keys.items():
        status = "[green]configured[/green]" if key else "[red]not set[/red]"
        table.add_row(source.capitalize(), status)
    console.print(table)


# ------------------------------------------------------------------ #
# search
# ------------------------------------------------------------------ #


@cli.command()
@click.argument("query")
@click.option("--sources", default="unsplash,pexels", show_default=True, help="Comma-separated source list")
@click.option("--count", default=100, show_default=True, type=int, help="Total images to download")
@click.option("--min-width", default=256, show_default=True, type=int)
@click.option("--min-height", default=256, show_default=True, type=int)
@click.option("--max-blur", default=100.0, show_default=True, type=float, help="Min sharpness (Laplacian variance)")
@click.option("--max-aspect-ratio", default=4.0, show_default=True, type=float)
@click.option("--output", default="./dataset", show_default=True, help="Output directory")
@click.option(
    "--format", "export_format",
    default="raw", show_default=True,
    type=click.Choice(FORMATS),
    help="Export format",
)
@click.option("--val-split", default=0.1, show_default=True, type=float, help="Validation split fraction")
@click.option("--test-split", default=0.0, show_default=True, type=float, help="Test split fraction")
@click.option("--label-by-query/--no-label-by-query", default=False, help="Put each query in its own subfolder")
@click.option("--dedup/--no-dedup", default=True, show_default=True, help="Enable perceptual deduplication")
@click.option("--clip-filter", is_flag=True, default=False, help="Filter by CLIP semantic similarity (requires [clip] extra)")
@click.option("--clip-threshold", default=0.2, show_default=True, type=float, help="CLIP cosine similarity threshold")
@click.option("--no-resume", is_flag=True, default=False, help="Re-download even if already in manifest")
@click.option("--workers", default=8, show_default=True, type=int, help="Parallel download workers")
@click.option("--failure-log", is_flag=True, default=False, help="Save download_failures.jsonl on completion")
def search(
    query: str,
    sources: str,
    count: int,
    min_width: int,
    min_height: int,
    max_blur: float,
    max_aspect_ratio: float,
    output: str,
    export_format: str,
    val_split: float,
    test_split: float,
    label_by_query: bool,
    dedup: bool,
    clip_filter: bool,
    clip_threshold: float,
    no_resume: bool,
    workers: int,
    failure_log: bool,
) -> None:
    """Search and download images from commercial-safe sources."""
    from .builder import DatasetBuilder
    from .config import get_api_key
    from .exporters.coco import COCOExporter
    from .exporters.csv_exporter import CSVExporter
    from .exporters.huggingface import HuggingFaceExporter
    from .exporters.raw import RawExporter
    from .exporters.yolo import YOLOExporter
    from .filters.dedup import Deduplicator
    from .filters.quality import QualityFilter
    from .sources.openimages import OpenImagesSource
    from .sources.pixabay import PixabaySource
    from .sources.pexels import PexelsSource
    from .sources.unsplash import UnsplashSource
    from .sources.wikimedia import WikimediaSource

    source_list = [s.strip().lower() for s in sources.split(",")]
    queries = [q.strip() for q in query.split(",")]
    output_dir = Path(output)

    builder = DatasetBuilder(output_dir, max_workers=workers)

    _source_factories = {
        "unsplash": lambda: UnsplashSource(get_api_key("unsplash") or ""),
        "pexels": lambda: PexelsSource(get_api_key("pexels") or ""),
        "pixabay": lambda: PixabaySource(get_api_key("pixabay") or ""),
        "wikimedia": lambda: WikimediaSource(),
        "openimages": lambda: OpenImagesSource(),
    }

    for src_name in source_list:
        if src_name not in _source_factories:
            console.print(f"[red]Unknown source: {src_name}. Choose from: {', '.join(SOURCES)}[/red]")
            continue
        src_obj = _source_factories[src_name]()
        if hasattr(src_obj, "api_key") and not src_obj.api_key:
            console.print(
                f"[yellow]No API key for {src_name}. "
                f"Set with: idc config set --{src_name}-key KEY[/yellow]"
            )
            continue
        builder.add_source(src_obj)

    if not builder._sources:
        console.print("[red]No valid sources configured.[/red]")
        raise click.Abort()

    builder.add_filter(
        QualityFilter(
            min_width=min_width,
            min_height=min_height,
            blur_threshold=max_blur,
            max_aspect_ratio=max_aspect_ratio,
        )
    )
    if dedup:
        builder.add_filter(Deduplicator())

    if clip_filter:
        from .filters.clip_filter import CLIPFilter
        builder.add_filter(CLIPFilter(threshold=clip_threshold))

    exporter_map = {
        "raw": lambda: RawExporter(val_split=val_split, test_split=test_split),
        "yolo": lambda: YOLOExporter(val_split=val_split, test_split=test_split),
        "coco": lambda: COCOExporter(val_split=val_split, test_split=test_split),
        "huggingface": lambda: HuggingFaceExporter(val_split=val_split, test_split=test_split),
        "csv": lambda: CSVExporter(),
    }

    total_accepted = 0
    for q in queries:
        console.print(f"\n[bold]Query:[/bold] {q}")
        records = builder.search(q, count)
        label = q.replace(" ", "_") if label_by_query else None
        accepted = builder.download(records, label=label, skip_existing=not no_resume, save_failure_log=failure_log)
        total_accepted += len(accepted)
        console.print(f"[green]{len(accepted)} images accepted for '{q}'[/green]")

    builder.export(exporter_map[export_format]())
    console.print(f"\n[bold green]Done! {total_accepted} images saved to {output_dir}[/bold green]")


# ------------------------------------------------------------------ #
# filter  (fixed: actually modifies manifest + deletes files)
# ------------------------------------------------------------------ #


@cli.command("filter")
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.option("--min-width", default=256, show_default=True, type=int)
@click.option("--min-height", default=256, show_default=True, type=int)
@click.option("--max-blur", default=100.0, show_default=True, type=float)
@click.option("--dedup/--no-dedup", default=False, help="Remove near-duplicates")
@click.option("--keep-files", is_flag=True, default=False, help="Don't delete rejected image files")
@click.option("--dry-run", is_flag=True, default=False, help="Report what would be removed without changing anything")
def filter_cmd(
    dataset_dir: str,
    min_width: int,
    min_height: int,
    max_blur: float,
    dedup: bool,
    keep_files: bool,
    dry_run: bool,
) -> None:
    """
    Apply quality filters to an already-downloaded dataset.

    Removes rejected records from the manifest and deletes their image files
    (use --keep-files to skip deletion, --dry-run to preview without changes).
    """
    from .filters.dedup import Deduplicator
    from .filters.quality import QualityFilter
    from .manifest import Manifest

    dataset_path = Path(dataset_dir)
    manifest = Manifest(dataset_path / "manifest.db")
    records = manifest.get_all()

    qf = QualityFilter(min_width=min_width, min_height=min_height, blur_threshold=max_blur)
    deduplicator = Deduplicator() if dedup else None

    to_remove: list[str] = []
    kept = 0

    for record in records:
        passed, reason = qf.check(record)
        if not passed:
            to_remove.append(record.id)
            if not dry_run and not keep_files and record.local_path:
                Path(record.local_path).unlink(missing_ok=True)
            continue

        if deduplicator:
            is_unique, _ = deduplicator.check_and_add(record)
            if not is_unique:
                to_remove.append(record.id)
                if not dry_run and not keep_files and record.local_path:
                    Path(record.local_path).unlink(missing_ok=True)
                continue

        kept += 1

    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] would remove {len(to_remove)}, keep {kept} "
            f"(use without --dry-run to apply)"
        )
    else:
        manifest.remove_many(to_remove)
        console.print(f"[green]Filter complete: {kept} kept, {len(to_remove)} removed from manifest[/green]")


# ------------------------------------------------------------------ #
# export
# ------------------------------------------------------------------ #


@cli.command("export")
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.option("--format", "export_format", default="raw", type=click.Choice(FORMATS), show_default=True)
@click.option("--output", default=None, help="Output directory (default: <dataset_dir>/export_<format>)")
@click.option("--val-split", default=0.1, show_default=True, type=float, help="Validation split fraction")
@click.option("--test-split", default=0.0, show_default=True, type=float, help="Test split fraction")
def export_cmd(
    dataset_dir: str,
    export_format: str,
    output: Optional[str],
    val_split: float,
    test_split: float,
) -> None:
    """Export a dataset to a different format."""
    from .exporters.coco import COCOExporter
    from .exporters.csv_exporter import CSVExporter
    from .exporters.huggingface import HuggingFaceExporter
    from .exporters.raw import RawExporter
    from .exporters.yolo import YOLOExporter
    from .manifest import Manifest

    dataset_path = Path(dataset_dir)
    manifest = Manifest(dataset_path / "manifest.db")
    records = manifest.get_all()

    output_dir = Path(output) if output else dataset_path / f"export_{export_format}"
    exporter = {
        "raw": lambda: RawExporter(val_split=val_split, test_split=test_split),
        "yolo": lambda: YOLOExporter(val_split=val_split, test_split=test_split),
        "coco": lambda: COCOExporter(val_split=val_split, test_split=test_split),
        "huggingface": lambda: HuggingFaceExporter(val_split=val_split, test_split=test_split),
        "csv": lambda: CSVExporter(),
    }[export_format]()
    exporter.export(records, output_dir)
    console.print(f"[green]Exported {len(records)} images to {output_dir}[/green]")


# ------------------------------------------------------------------ #
# report
# ------------------------------------------------------------------ #


@cli.command("report")
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.option("--output", default="attribution.csv", show_default=True, help="Output CSV path")
def report(dataset_dir: str, output: str) -> None:
    """Generate an attribution/license report for a dataset."""
    from .manifest import Manifest

    dataset_path = Path(dataset_dir)
    manifest = Manifest(dataset_path / "manifest.db")
    records = manifest.get_all()

    output_path = Path(output)
    fieldnames = [
        "id", "attribution", "license_type", "license_url",
        "photographer", "photographer_url", "url", "source", "query", "local_path",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "id": r.id,
                    "attribution": r.attribution,
                    "license_type": r.license_type,
                    "license_url": r.license_url,
                    "photographer": r.photographer,
                    "photographer_url": r.photographer_url,
                    "url": r.url,
                    "source": r.source,
                    "query": r.query,
                    "local_path": str(r.local_path) if r.local_path else "",
                }
            )

    console.print(f"[green]Attribution report: {output_path} ({len(records)} entries)[/green]")


# ------------------------------------------------------------------ #
# merge
# ------------------------------------------------------------------ #


@cli.command("merge")
@click.argument("source_a", type=click.Path(exists=True))
@click.argument("source_b", type=click.Path(exists=True))
@click.option("--output", required=True, help="Output directory for merged dataset")
@click.option("--dedup/--no-dedup", default=True, show_default=True, help="Cross-dataset deduplication")
def merge(source_a: str, source_b: str, output: str, dedup: bool) -> None:
    """Merge two datasets, optionally deduplicating across them."""
    import shutil

    from .filters.dedup import Deduplicator
    from .manifest import Manifest

    src_a = Path(source_a)
    src_b = Path(source_b)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_a = Manifest(src_a / "manifest.db")
    manifest_b = Manifest(src_b / "manifest.db")
    records_a = manifest_a.get_all()
    records_b = manifest_b.get_all()

    console.print(f"[dim]Dataset A: {len(records_a)} images[/dim]")
    console.print(f"[dim]Dataset B: {len(records_b)} images[/dim]")

    deduplicator = Deduplicator() if dedup else None
    out_manifest = Manifest(out_dir / "manifest.db")
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    accepted = skipped_dup = skipped_missing = 0

    for record in records_a + records_b:
        if not record.local_path or not Path(record.local_path).exists():
            skipped_missing += 1
            continue

        if deduplicator:
            is_unique, _ = deduplicator.check_and_add(record)
            if not is_unique:
                skipped_dup += 1
                continue

        src_file = Path(record.local_path)
        dest_file = images_dir / src_file.name
        if dest_file.exists() and dest_file != src_file:
            dest_file = images_dir / f"{record.source}_{record.source_id}{src_file.suffix}"

        if src_file != dest_file:
            shutil.copy2(src_file, dest_file)

        record.local_path = dest_file
        out_manifest.add(record)
        accepted += 1

    console.print(
        f"[green]Merge complete: {accepted} images accepted, "
        f"{skipped_dup} duplicates removed, {skipped_missing} missing files skipped[/green]"
    )
    console.print(f"[green]Output: {out_dir}[/green]")


# ------------------------------------------------------------------ #
# info
# ------------------------------------------------------------------ #


@cli.command("info")
@click.argument("dataset_dir", type=click.Path(exists=True))
def info(dataset_dir: str) -> None:
    """Show statistics for a dataset."""
    from .manifest import Manifest

    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "manifest.db"
    if not manifest_path.exists():
        console.print(f"[red]No manifest found at {manifest_path}[/red]")
        raise click.Abort()

    manifest = Manifest(manifest_path)
    records = manifest.get_all()

    if not records:
        console.print("[yellow]Dataset is empty.[/yellow]")
        return

    by_source = Counter(r.source for r in records)
    by_license = Counter(r.license_type for r in records)
    widths = [r.width for r in records if r.width]
    heights = [r.height for r in records if r.height]

    table = Table(title=f"Dataset Info: {dataset_dir}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total images", str(len(records)))
    table.add_row("Sources", ", ".join(f"{k}: {v}" for k, v in by_source.most_common()))
    table.add_row("Licenses", ", ".join(f"{k}: {v}" for k, v in by_license.most_common()))
    if widths:
        table.add_row("Avg width", f"{sum(widths) / len(widths):.0f}px")
        table.add_row("Avg height", f"{sum(heights) / len(heights):.0f}px")
        table.add_row("Min width", f"{min(widths)}px")
        table.add_row("Min height", f"{min(heights)}px")

    console.print(table)
