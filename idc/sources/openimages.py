from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import List, Optional

import httpx
from rich.console import Console

from ..models import ImageRecord
from ..utils import retry_request
from .base import ImageSource

console = Console()

_DESCRIPTIONS_URL = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"

_IMAGES_URLS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images.csv",
}

_LABELS_URLS = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
}

_APPROX_SIZES = {
    "validation": "~28 MB",
    "test": "~28 MB",
    "train": "~250 MB",
}


class OpenImagesSource(ImageSource):
    """
    Open Images V6 dataset source (no API key required).

    On first use, metadata CSV files are downloaded and indexed into a local
    SQLite cache at ~/.idc/openimages/. Subsequent searches are instant.

    Splits:
        validation  — ~41K images, ~28 MB download (default)
        test        — ~125K images, ~28 MB download
        train       — ~1.7M images, ~250 MB download (opt-in)

    Licenses:
        Most images are CC BY 2.0 (attribution required).
        License info per image is included in the metadata.
    """

    name = "openimages"

    def __init__(
        self,
        split: str = "validation",
        cache_dir: Optional[Path] = None,
    ) -> None:
        if split not in _IMAGES_URLS:
            raise ValueError(f"split must be one of {list(_IMAGES_URLS)}; got {split!r}")
        self.split = split
        self.cache_dir = Path(cache_dir or Path.home() / ".idc" / "openimages")
        self._db: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        if self._db is not None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.cache_dir / f"index_{self.split}.db"
        self._db = sqlite3.connect(str(db_path))

        already_indexed = self._db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='images'"
        ).fetchone()[0]

        if not already_indexed:
            self._build_index()

    def _build_index(self) -> None:
        console.print(
            f"[bold]Open Images[/bold] — building {self.split} index "
            f"(first run, download ~{_APPROX_SIZES[self.split]})…"
        )

        self._db.executescript(  # type: ignore[union-attr]
            """
            CREATE TABLE IF NOT EXISTS classes (
                label_name   TEXT PRIMARY KEY,
                display_name TEXT
            );
            CREATE TABLE IF NOT EXISTS images (
                image_id          TEXT PRIMARY KEY,
                original_url      TEXT,
                thumbnail_url     TEXT,
                author            TEXT,
                author_url        TEXT,
                license_url       TEXT,
                title             TEXT
            );
            CREATE TABLE IF NOT EXISTS image_labels (
                image_id   TEXT,
                label_name TEXT,
                PRIMARY KEY (image_id, label_name)
            );
            CREATE INDEX IF NOT EXISTS idx_labels ON image_labels(label_name);
            """
        )

        with httpx.Client(timeout=300, follow_redirects=True) as client:
            self._download_classes(client)
            self._download_images(client)
            self._download_labels(client)

        self._db.commit()  # type: ignore[union-attr]
        console.print(f"[green]Open Images {self.split} index ready.[/green]")

    def _stream_csv(self, client: httpx.Client, url: str, dest: Path) -> None:
        if dest.exists():
            return
        console.print(f"[dim]  Downloading {dest.name}…[/dim]")
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

    def _download_classes(self, client: httpx.Client) -> None:
        dest = self.cache_dir / "class-descriptions.csv"
        self._stream_csv(client, _DESCRIPTIONS_URL, dest)
        with open(dest, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self._db.executemany(  # type: ignore[union-attr]
            "INSERT OR IGNORE INTO classes VALUES (?, ?)", rows
        )

    def _download_images(self, client: httpx.Client) -> None:
        dest = self.cache_dir / f"{self.split}-images.csv"
        self._stream_csv(client, _IMAGES_URLS[self.split], dest)

        batch: list = []
        with open(dest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch.append(
                    (
                        row.get("ImageID", ""),
                        row.get("OriginalURL", ""),
                        row.get("Thumbnail300KURL", ""),
                        row.get("Author", ""),
                        row.get("AuthorProfileURL", ""),
                        row.get("License", ""),
                        row.get("Title", ""),
                    )
                )
                if len(batch) >= 2000:
                    self._db.executemany(  # type: ignore[union-attr]
                        "INSERT OR IGNORE INTO images VALUES (?,?,?,?,?,?,?)", batch
                    )
                    batch = []
        if batch:
            self._db.executemany(  # type: ignore[union-attr]
                "INSERT OR IGNORE INTO images VALUES (?,?,?,?,?,?,?)", batch
            )

    def _download_labels(self, client: httpx.Client) -> None:
        dest = self.cache_dir / f"{self.split}-labels.csv"
        self._stream_csv(client, _LABELS_URLS[self.split], dest)

        batch: list = []
        with open(dest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Confidence 1 = human-verified positive
                if row.get("Confidence", "0") == "1":
                    batch.append((row.get("ImageID", ""), row.get("LabelName", "")))
                if len(batch) >= 5000:
                    self._db.executemany(  # type: ignore[union-attr]
                        "INSERT OR IGNORE INTO image_labels VALUES (?,?)", batch
                    )
                    batch = []
        if batch:
            self._db.executemany(  # type: ignore[union-attr]
                "INSERT OR IGNORE INTO image_labels VALUES (?,?)", batch
            )

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(self, query: str, count: int, **kwargs) -> List[ImageRecord]:
        self._ensure_index()

        # Find matching class label names (case-insensitive substring match)
        class_ids = [
            row[0]
            for row in self._db.execute(  # type: ignore[union-attr]
                "SELECT label_name FROM classes WHERE LOWER(display_name) LIKE ?",
                (f"%{query.lower()}%",),
            ).fetchall()
        ]

        if not class_ids:
            console.print(f"[yellow]Open Images: no classes found matching '{query}'[/yellow]")
            return []

        placeholders = ",".join("?" * len(class_ids))
        rows = self._db.execute(  # type: ignore[union-attr]
            f"""
            SELECT DISTINCT i.image_id, i.original_url, i.thumbnail_url,
                            i.author, i.author_url, i.license_url, i.title
            FROM images i
            JOIN image_labels il ON i.image_id = il.image_id
            WHERE il.label_name IN ({placeholders})
              AND i.original_url != ''
            LIMIT ?
            """,
            class_ids + [count],
        ).fetchall()

        return [self._to_record(row, query) for row in rows]

    def _license_from_url(self, url: str) -> str:
        u = url.lower()
        if "cc0" in u or "publicdomain" in u:
            return "CC0-1.0"
        if "by/4.0" in u:
            return "CC-BY-4.0"
        if "by/2.0" in u or "by/2" in u:
            return "CC-BY-2.0"
        return "CC-BY-2.0"  # Open Images default

    def _to_record(self, row: tuple, query: str) -> ImageRecord:
        image_id, original_url, thumbnail_url, author, author_url, license_url, title = row
        license_type = self._license_from_url(license_url)
        attribution = (
            f"Photo by {author} (Open Images, {license_type})"
            if author
            else f"Open Images Dataset ({license_type})"
        )
        return ImageRecord(
            source="openimages",
            source_id=image_id,
            url=f"https://www.flickr.com/photos/{image_id}",
            download_url=original_url,
            license_type=license_type,
            license_url=license_url,
            photographer=author,
            photographer_url=author_url,
            attribution=attribution,
            width=0,
            height=0,
            tags=[query],
            description=title,
            query=query,
        )

    async def adownload(self, record: ImageRecord, output_dir: Path) -> Path:
        """Async download with fallback to thumbnail if original URL fails."""
        import httpx as _httpx

        from ..utils import async_retry_request

        output_dir.mkdir(parents=True, exist_ok=True)
        ext = self._guess_extension(record.download_url)
        local_path = output_dir / f"openimages_{record.source_id}{ext}"

        if local_path.exists():
            return local_path

        async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                resp = await async_retry_request(client, "GET", record.download_url, max_retries=2)
                local_path.write_bytes(resp.content)
            except Exception:
                if self._db:
                    row = self._db.execute(
                        "SELECT thumbnail_url FROM images WHERE image_id = ?",
                        (record.source_id,),
                    ).fetchone()
                    if row and row[0]:
                        resp = await async_retry_request(client, "GET", row[0], max_retries=2)
                        local_path.write_bytes(resp.content)
                    else:
                        raise
                else:
                    raise

        return local_path

    def download(self, record: ImageRecord, output_dir: Path) -> Path:
        """Download with fallback to thumbnail if original URL fails."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = self._guess_extension(record.download_url)
        local_path = output_dir / f"openimages_{record.source_id}{ext}"

        if local_path.exists():
            return local_path

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            try:
                resp = retry_request(client, "GET", record.download_url, max_retries=2)
                local_path.write_bytes(resp.content)
            except Exception:
                # Fallback: look up thumbnail URL from index
                if self._db:
                    row = self._db.execute(
                        "SELECT thumbnail_url FROM images WHERE image_id = ?",
                        (record.source_id,),
                    ).fetchone()
                    if row and row[0]:
                        resp = retry_request(client, "GET", row[0], max_retries=2)
                        local_path.write_bytes(resp.content)
                    else:
                        raise
                else:
                    raise

        return local_path
