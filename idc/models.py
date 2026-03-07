from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    source_id: str
    url: str
    download_url: str
    local_path: Optional[Path] = None

    # License
    license_type: str
    license_url: str
    attribution: str
    photographer: str
    photographer_url: str

    # Metadata
    width: int
    height: int
    file_size_bytes: int = 0
    format: str = ""
    tags: List[str] = Field(default_factory=list)
    description: str = ""

    # Quality signals
    blur_score: Optional[float] = None
    phash: Optional[str] = None

    # Provenance
    downloaded_at: datetime = Field(default_factory=datetime.utcnow)
    query: str = ""
    dataset_name: str = ""

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["local_path"] = str(d["local_path"]) if d["local_path"] else None
        if isinstance(d["downloaded_at"], datetime):
            d["downloaded_at"] = d["downloaded_at"].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ImageRecord:
        data = dict(d)
        if "downloaded_at" in data and isinstance(data["downloaded_at"], str):
            data["downloaded_at"] = datetime.fromisoformat(data["downloaded_at"])
        return cls(**data)
