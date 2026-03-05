"""Unified detection + segmentation using SAM3 (Segment Anything Model 3).

SAM3 takes text prompts and directly outputs bounding boxes + instance masks,
replacing both Grounding DINO (detector) and SAM2 (segmentor) in one model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import PIL.Image

from perception.detector import Detection

logger = logging.getLogger(__name__)


class Sam3Detector:
    """Detect and segment objects using SAM3 with text prompts.

    SAM3 performs Promptable Concept Segmentation — given a text concept
    (e.g. "kitchen sink"), it finds all matching instances and returns both
    bounding boxes and pixel-level masks.

    The model is lazy-loaded on the first call.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    confidence_threshold:
        Minimum score for a detection to be kept.
    mask_threshold:
        Threshold for binarising predicted masks.
    """

    MODEL_ID = "facebook/sam3"

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        """Load the SAM3 model and processor if not already loaded."""
        if self._model is not None:
            return

        from transformers import Sam3Model, Sam3Processor

        logger.info("Loading SAM3 from %s …", self.MODEL_ID)
        self._processor = Sam3Processor.from_pretrained(self.MODEL_ID)
        self._model = Sam3Model.from_pretrained(self.MODEL_ID).to(self.device)
        self._model.eval()
        logger.info("SAM3 loaded on %s", self.device)

    def detect_and_segment(
        self,
        image: PIL.Image.Image,
        tags: list[str],
        iou_threshold: float = 0.5,
        return_pre_nms: bool = False,
    ) -> tuple[list[Detection], list[np.ndarray]] | tuple[list[Detection], list[np.ndarray], list[Detection]]:
        """Detect and segment objects for each tag using SAM3.

        Runs SAM3 once per tag (each tag is a text concept prompt), then
        applies cross-category NMS to remove duplicate detections.

        Parameters
        ----------
        image:
            An RGB PIL image.
        tags:
            List of object labels (e.g. ``["couch", "kitchen sink", "lamp"]``).
        iou_threshold:
            IoU threshold for cross-category NMS.
        return_pre_nms:
            When True, also return pre-NMS detections as a third element.

        Returns
        -------
        tuple
            ``(detections, masks)`` or ``(detections, masks, pre_nms_detections)``
            if *return_pre_nms* is True.
        """
        self._ensure_model()
        image = image.convert("RGB")
        image_size = image.size  # (width, height)

        all_detections: list[Detection] = []
        all_masks: list[np.ndarray] = []

        for tag in tags:
            tag_clean = tag.strip()
            logger.debug("SAM3 detecting: '%s'", tag_clean)

            inputs = self._processor(
                images=image, text=tag_clean, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results.get("masks", [])
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])

            for i in range(len(masks)):
                mask = masks[i]
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                mask = mask.astype(bool)

                box = boxes[i]
                if torch.is_tensor(box):
                    box = box.tolist()
                else:
                    box = list(box)

                score = float(scores[i]) if torch.is_tensor(scores[i]) else float(scores[i])

                det = Detection(
                    bbox=box,
                    label=tag_clean,
                    confidence=score,
                )
                all_detections.append(det)
                all_masks.append(mask)

            logger.debug("  Tag '%s': %d instances", tag_clean, len(masks))

        logger.info(
            "SAM3 per-tag detection: %d tags → %d raw detections",
            len(tags), len(all_detections),
        )

        if not all_detections:
            if return_pre_nms:
                return [], [], []
            return [], []

        # Cross-category NMS
        pre_nms_detections = list(all_detections) if return_pre_nms else None

        from perception.nms import cross_category_nms

        filtered_detections = cross_category_nms(all_detections, iou_threshold=iou_threshold)

        # Build index of which detections survived NMS
        # Match by bbox since Detection is a dataclass
        surviving_indices = []
        for fd in filtered_detections:
            for i, od in enumerate(all_detections):
                if (
                    od.bbox == fd.bbox
                    and od.label == fd.label
                    and abs(od.confidence - fd.confidence) < 1e-6
                    and i not in surviving_indices
                ):
                    surviving_indices.append(i)
                    break

        filtered_masks = [all_masks[i] for i in surviving_indices]

        logger.info(
            "SAM3 after NMS: %d detections",
            len(filtered_detections),
        )

        if return_pre_nms:
            return filtered_detections, filtered_masks, pre_nms_detections
        return filtered_detections, filtered_masks
