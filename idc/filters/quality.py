from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from ..models import ImageRecord

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class QualityFilter:
    """
    Filters images by resolution, aspect ratio, file size, and sharpness.

    blur_threshold: minimum Laplacian variance required. Images scoring below
    this are considered too blurry and rejected. Default 100.
    """

    def __init__(
        self,
        min_width: int = 256,
        min_height: int = 256,
        max_aspect_ratio: float = 4.0,
        min_file_size: int = 10 * 1024,
        blur_threshold: Optional[float] = 100.0,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.min_file_size = min_file_size
        self.blur_threshold = blur_threshold

    def check(self, record: ImageRecord) -> Tuple[bool, str]:
        """Return (passed, reason). reason is empty string on success."""
        if not record.local_path:
            return False, "no local file"
        local_path = Path(record.local_path)
        if not local_path.exists():
            return False, "file not found"

        if local_path.stat().st_size < self.min_file_size:
            return False, f"file too small ({local_path.stat().st_size} < {self.min_file_size})"

        try:
            img = Image.open(local_path)
            img.verify()
            img = Image.open(local_path)  # reopen after verify
        except Exception as exc:
            return False, f"invalid image: {exc}"

        w, h = img.size
        if w < self.min_width:
            return False, f"width {w} < {self.min_width}"
        if h < self.min_height:
            return False, f"height {h} < {self.min_height}"

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > self.max_aspect_ratio:
            return False, f"aspect ratio {aspect:.2f} > {self.max_aspect_ratio}"

        if self.blur_threshold is not None:
            blur = record.blur_score if record.blur_score is not None else self._compute_blur(local_path, img)
            if blur is not None and blur < self.blur_threshold:
                return False, f"too blurry (score={blur:.1f} < threshold={self.blur_threshold})"

        return True, ""

    def compute_quality_signals(self, record: ImageRecord, img: Optional[Image.Image] = None) -> dict:
        """Compute quality signals for a record. Updates record fields in place."""
        if not record.local_path:
            return {}
        local_path = Path(record.local_path)
        if not local_path.exists():
            return {}

        signals: dict = {"file_size_bytes": local_path.stat().st_size}

        if img is None:
            try:
                img = Image.open(local_path)
            except Exception:
                return signals

        signals["width"] = img.size[0]
        signals["height"] = img.size[1]
        signals["format"] = img.format or ""

        blur = self._compute_blur(local_path, img)
        if blur is not None:
            signals["blur_score"] = blur

        return signals

    def _compute_blur(self, path: Path, img: Image.Image) -> Optional[float]:
        if _HAS_CV2:
            try:
                import numpy as np

                img_cv = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img_cv is not None:
                    return float(cv2.Laplacian(img_cv, cv2.CV_64F).var())
            except Exception:
                pass

        # Fallback: numpy variance on grayscale
        try:
            import numpy as np

            arr = np.array(img.convert("L"), dtype=float)
            return float(arr.var())
        except Exception:
            return None
