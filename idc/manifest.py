from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import ImageRecord


class Manifest:
    """SQLite-backed manifest tracking every image in the dataset."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id        TEXT PRIMARY KEY,
                    source    TEXT NOT NULL,
                    source_id TEXT,
                    query     TEXT NOT NULL,
                    phash     TEXT,
                    data      TEXT NOT NULL
                )
                """
            )
            # Migration: add source_id column to existing databases
            try:
                conn.execute("ALTER TABLE images ADD COLUMN source_id TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

            conn.execute("CREATE INDEX IF NOT EXISTS idx_phash ON images(phash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON images(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_id ON images(source, source_id)")
            conn.commit()

    def add(self, record: ImageRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO images (id, source, source_id, query, phash, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.source,
                    record.source_id,
                    record.query,
                    record.phash,
                    json.dumps(record.to_dict()),
                ),
            )
            conn.commit()

    def get_all(self) -> List[ImageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT data FROM images").fetchall()
        return [ImageRecord.from_dict(json.loads(row[0])) for row in rows]

    def get_phashes(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT phash FROM images WHERE phash IS NOT NULL"
            ).fetchall()
        return [row[0] for row in rows]

    def get_by_source(self, source: str) -> List[ImageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM images WHERE source = ?", (source,)
            ).fetchall()
        return [ImageRecord.from_dict(json.loads(row[0])) for row in rows]

    def get_by_id(self, record_id: str) -> Optional[ImageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM images WHERE id = ?", (record_id,)
            ).fetchone()
        return ImageRecord.from_dict(json.loads(row[0])) if row else None

    def get_by_source_id(self, source: str, source_id: str) -> Optional[ImageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM images WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        return ImageRecord.from_dict(json.loads(row[0])) if row else None

    def has_source_id(self, source: str, source_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM images WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        return row is not None

    def remove(self, record_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM images WHERE id = ?", (record_id,))
            conn.commit()

    def remove_many(self, record_ids: List[str]) -> int:
        """Remove multiple records by id. Returns number removed."""
        if not record_ids:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(record_ids))
            cursor = conn.execute(
                f"DELETE FROM images WHERE id IN ({placeholders})", record_ids
            )
            conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
