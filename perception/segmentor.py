"""Instance segmentation using SAM2 with bbox prompts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import PIL.Image

    from perception.detector import Detection

logger = logging.getLogger(__name__)


class Segmentor:
    """Segment objects using SAM with detected bboxes as prompts.

    Attempts to load ``facebook/sam2-hiera-large`` first.  If that model is
    not available (e.g. missing ``sam2`` extras) it falls back to the original
    ``facebook/sam-vit-large`` which ships with the ``transformers`` library.

    Models are lazy-loaded on the first call to ``segment()``.
    """

    SAM2_MODEL = "facebook/sam2-hiera-large"
    SAM_FALLBACK_MODEL = "facebook/sam-vit-large"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._backend: str | None = None  # "sam2" or "sam"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_sam2(self) -> bool:
        """Try to load SAM2 via the transformers ``SamModel``/``SamProcessor`` API."""
        try:
            from transformers import Sam2ForImageSegmentation, Sam2Processor

            logger.info("Loading SAM2 from %s …", self.SAM2_MODEL)
            self._processor = Sam2Processor.from_pretrained(self.SAM2_MODEL)
            self._model = Sam2ForImageSegmentation.from_pretrained(self.SAM2_MODEL).to(
                self.device
            )
            self._model.eval()
            self._backend = "sam2"
            logger.info("SAM2 loaded successfully on %s", self.device)
            return True
        except Exception:
            logger.warning("SAM2 not available, will try SAM fallback.")
            return False

    def _load_sam_fallback(self) -> bool:
        """Load the original SAM model as a fallback."""
        try:
            from transformers import SamModel, SamProcessor

            logger.info("Loading SAM from %s …", self.SAM_FALLBACK_MODEL)
            self._processor = SamProcessor.from_pretrained(self.SAM_FALLBACK_MODEL)
            self._model = SamModel.from_pretrained(self.SAM_FALLBACK_MODEL).to(self.device)
            self._model.eval()
            self._backend = "sam"
            logger.info("SAM loaded successfully on %s", self.device)
            return True
        except Exception:
            logger.error("Failed to load SAM fallback model.")
            return False

    def _ensure_model(self) -> None:
        """Load a segmentation model if one has not been loaded yet."""
        if self._model is not None:
            return
        if not self._load_sam2() and not self._load_sam_fallback():
            raise RuntimeError(
                "Could not load any segmentation model (tried SAM2 and SAM)."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _segment_single(
        self, image: PIL.Image.Image, bbox: list[float]
    ) -> np.ndarray:
        """Segment a single object defined by *bbox* and return a (H, W) bool mask."""
        # SAM / SAM2 processors expect bboxes as [[x_min, y_min, x_max, y_max]]
        input_boxes = [[[bbox]]]  # batch of 1 image, 1 point set, 1 box

        inputs = self._processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process to get masks at original resolution
        masks = self._processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )

        # masks is a list (batch) of tensors shaped (num_prompts, num_masks, H, W)
        # Take the first batch element, first prompt, and pick the best mask
        # (SAM produces multiple mask proposals ranked by predicted IoU)
        mask_tensor = masks[0]  # (num_prompts, num_masks, H, W)
        if mask_tensor.dim() == 4:
            # Multiple masks per prompt — pick the one with highest IoU score
            iou_scores = outputs.iou_scores[0]  # (num_prompts, num_masks)
            best_idx = iou_scores[0].argmax().item()
            mask_np: np.ndarray = mask_tensor[0, best_idx].cpu().numpy().astype(bool)
        elif mask_tensor.dim() == 3:
            mask_np = mask_tensor[0, 0].cpu().numpy().astype(bool)
        else:
            mask_np = mask_tensor.cpu().numpy().astype(bool)

        return mask_np

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        image: PIL.Image.Image,
        detections: list[Detection],
    ) -> list[np.ndarray]:
        """Produce a binary mask for each detection.

        Parameters
        ----------
        image:
            An RGB PIL image.
        detections:
            Detections from :class:`ObjectDetector`.

        Returns
        -------
        list[np.ndarray]
            Boolean masks of shape ``(H, W)`` at the original image resolution,
            one per detection.
        """
        self._ensure_model()

        image = image.convert("RGB")

        if not detections:
            logger.info("No detections provided — returning empty mask list.")
            return []

        masks: list[np.ndarray] = []
        for det in detections:
            try:
                mask = self._segment_single(image, det.bbox)
                masks.append(mask)
            except Exception:
                logger.warning(
                    "Segmentation failed for detection '%s' — returning empty mask.",
                    det.label,
                    exc_info=True,
                )
                # Return an all-False mask at the correct resolution
                w, h = image.size
                masks.append(np.zeros((h, w), dtype=bool))

        logger.info("Segmented %d / %d detections.", len(masks), len(detections))
        return masks
