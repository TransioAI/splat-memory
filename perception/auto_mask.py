"""SAM2 automatic mask generation (segment-everything, no prompts)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)


class AutoMaskGenerator:
    """Generate masks for all objects in an image using SAM2 auto-segmentation.

    Unlike the bbox-prompted ``Segmentor``, this runs SAM2's automatic mask
    generator which discovers and segments every object without text prompts.

    The model is lazy-loaded on the first call to ``generate()``.
    """

    MODEL_ID = "facebook/sam2-hiera-large"

    def __init__(
        self,
        device: str = "cuda",
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.6,
        score_threshold: float = 0.7,
        max_masks: int = 50,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.score_threshold = score_threshold
        self.max_masks = max_masks
        self._pipe = None

    def _ensure_model(self) -> None:
        """Load the mask-generation pipeline if not already loaded."""
        if self._pipe is not None:
            return

        from transformers import pipeline

        logger.info("Loading SAM2 auto-mask generator from %s …", self.MODEL_ID)
        self._pipe = pipeline(
            "mask-generation",
            model=self.MODEL_ID,
            device=self.device,
            dtype=torch.float32,
        )
        logger.info("SAM2 auto-mask generator loaded on %s", self.device)

    def generate(
        self, image: PIL.Image.Image
    ) -> tuple[list[np.ndarray], list[float]]:
        """Auto-segment the image and return filtered masks + scores.

        Parameters
        ----------
        image:
            An RGB PIL image.

        Returns
        -------
        tuple[list[np.ndarray], list[float]]
            ``(masks, scores)`` where each mask is a ``(H, W)`` bool array
            and scores are the predicted IoU values.  Sorted by score
            descending, filtered by area and score thresholds, capped at
            ``max_masks``.
        """
        self._ensure_model()
        image = image.convert("RGB")
        w, h = image.size
        total_pixels = w * h

        logger.info("Running SAM2 auto-mask generation on %dx%d image …", w, h)
        outputs = self._pipe(image, points_per_batch=64)

        raw_masks = outputs["masks"]
        raw_scores = outputs["scores"]
        logger.info("SAM2 generated %d raw masks", len(raw_masks))

        # Filter by score, then area
        kept_masks: list[np.ndarray] = []
        kept_scores: list[float] = []

        for mask, score in zip(raw_masks, raw_scores, strict=False):
            if score < self.score_threshold:
                continue

            mask_np = np.asarray(mask, dtype=bool)
            area_frac = mask_np.sum() / total_pixels

            if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
                continue

            kept_masks.append(mask_np)
            kept_scores.append(float(score))

        # Sort by score descending
        if kept_masks:
            order = sorted(range(len(kept_scores)), key=lambda i: kept_scores[i], reverse=True)
            kept_masks = [kept_masks[i] for i in order]
            kept_scores = [kept_scores[i] for i in order]

        # Cap at max_masks
        kept_masks = kept_masks[: self.max_masks]
        kept_scores = kept_scores[: self.max_masks]

        logger.info(
            "Auto-mask: %d masks after filtering (score>=%.2f, area [%.3f, %.2f])",
            len(kept_masks),
            self.score_threshold,
            self.min_area_frac,
            self.max_area_frac,
        )
        return kept_masks, kept_scores
