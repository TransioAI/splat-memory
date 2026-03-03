"""Open-vocabulary image tagging using RAM++ (Recognize Anything Model Plus)."""

from __future__ import annotations

import logging

import PIL.Image
import torch

logger = logging.getLogger(__name__)

RAM_PLUS_REPO = "xinyu1205/recognize-anything-plus-model"
RAM_PLUS_FILENAME = "ram_plus_swin_large_14m.pth"


class ImageTagger:
    """Tag an image with open-vocabulary labels using RAM++.

    The model is lazy-loaded on the first call to :meth:`tag`.
    Weights are downloaded from HuggingFace Hub on first use.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    image_size:
        Input resolution for RAM++ (default 384).
    """

    def __init__(self, device: str = "cuda", image_size: int = 384) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self._model = None
        self._transform = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load RAM++ model and transform if not already loaded."""
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download
        from ram.models import ram_plus
        from ram.transform import get_transform

        logger.info("Downloading RAM++ weights from %s …", RAM_PLUS_REPO)
        weight_path = hf_hub_download(
            repo_id=RAM_PLUS_REPO,
            filename=RAM_PLUS_FILENAME,
        )
        logger.info("RAM++ weights at: %s", weight_path)

        logger.info("Loading RAM++ model (ViT=swin_l, image_size=%d) …", self.image_size)
        self._model = ram_plus(
            pretrained=weight_path,
            image_size=self.image_size,
            vit="swin_l",
        )
        self._model.eval()
        self._model = self._model.to(self.device)

        self._transform = get_transform(image_size=self.image_size)
        logger.info("RAM++ loaded successfully on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, image: PIL.Image.Image) -> list[str]:
        """Tag an image and return a list of predicted tag strings.

        Parameters
        ----------
        image:
            An RGB PIL image.

        Returns
        -------
        list[str]
            Raw tags predicted by RAM++.  Tags are unfiltered and may include
            hypernyms, scene labels, materials, etc.
        """
        self._ensure_model()

        from ram.inference import inference_ram

        image = image.convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        tags_en, _tags_zh = inference_ram(tensor, self._model)

        # tags_en is a pipe/comma-separated string of English tags
        separator = "|" if "|" in tags_en else ","
        tags = [t.strip() for t in tags_en.split(separator) if t.strip()]

        logger.info("RAM++ produced %d raw tags: %s", len(tags), tags)
        return tags
