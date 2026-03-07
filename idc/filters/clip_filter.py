from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from ..models import ImageRecord


class CLIPFilter:
    """
    Filters images by semantic similarity to the search query using CLIP.

    Images whose cosine similarity to the query text falls below `threshold`
    are rejected. Fail-open: images that cannot be scored (e.g. decode error)
    are accepted rather than rejected.

    Requires: pip install image-dataset-creator[clip]
    (open_clip_torch, torch, torchvision)
    """

    def __init__(
        self,
        threshold: float = 0.2,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device: Optional[str] = None
        self._text_cache: Dict[str, object] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "CLIP filtering requires open_clip_torch and torch.\n"
                "Install with: pip install image-dataset-creator[clip]"
            ) from exc

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model.eval()

    def _get_text_features(self, query: str) -> object:
        import torch

        if query not in self._text_cache:
            tokens = self._tokenizer([query]).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            self._text_cache[query] = features
        return self._text_cache[query]

    def check(self, record: ImageRecord) -> Tuple[bool, str]:
        """Return (passed, reason). Fail-open on errors."""
        if not record.local_path:
            return False, "no local file"
        local_path = Path(record.local_path)
        if not local_path.exists():
            return False, "file not found"

        try:
            self._load_model()
        except ImportError as exc:
            return True, ""  # CLIP not installed — pass through

        try:
            import torch
            from PIL import Image

            img = Image.open(local_path).convert("RGB")
            img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                img_features = self._model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            text_features = self._get_text_features(record.query or "")
            similarity = float((img_features @ text_features.T).item())

            if similarity < self.threshold:
                return False, (
                    f"CLIP similarity {similarity:.3f} < threshold {self.threshold}"
                )
            return True, ""
        except Exception:
            return True, ""  # fail open on decode/inference errors
