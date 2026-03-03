"""Orchestrates the full perception pipeline: detect -> segment -> depth."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image

from perception.depth import DepthEstimator
from perception.detector import Detection, ObjectDetector
from perception.segmentor import Segmentor

logger = logging.getLogger(__name__)

# Architectural / structural elements that are critical spatial anchors for
# scene-graph reasoning.  These are always sent to Grounding DINO regardless of
# whether RAM++ detected them.  If DINO finds 0 boxes for a tag it simply drops
# out — no false positives are forced into the scene graph.
SPATIAL_ANCHORS = {"wall", "ceiling", "floor", "door", "window", "countertop", "shelf"}


@dataclass
class PerceptionResult:
    """Output of the full perception pipeline."""

    detections: list[Detection]
    masks: list[np.ndarray]
    depth_map: np.ndarray
    image_size: tuple[int, int]  # (width, height)

    @property
    def num_objects(self) -> int:
        return len(self.detections)

    @staticmethod
    def empty(image_size: tuple[int, int]) -> PerceptionResult:
        """Create an empty result for when no objects are detected."""
        w, h = image_size
        return PerceptionResult(
            detections=[],
            masks=[],
            depth_map=np.zeros((h, w), dtype=np.float32),
            image_size=image_size,
        )


class PerceptionPipeline:
    """Run detection, segmentation, and depth sequentially on one image.

    Each sub-model is lazy-loaded on its first use.

    When ``use_tagger`` is True (default) and no explicit ``text_prompts``
    are provided, the pipeline runs RAM++ to discover objects, Claude to
    filter tags, and Grounding DINO one tag at a time for best accuracy.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.  Automatically falls back to CPU if CUDA
        is unavailable.
    confidence_threshold:
        Minimum score for a detection to be kept.
    use_tagger:
        When True, use RAM++ → Claude filter → per-tag DINO instead of the
        default all-at-once prompt.  Ignored when ``text_prompts`` is provided.
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        use_tagger: bool = True,
    ) -> None:
        self.detector = ObjectDetector(device=device, confidence_threshold=confidence_threshold)
        self.segmentor = Segmentor(device=device)
        self.depth_estimator = DepthEstimator(device=device)
        self.use_tagger = use_tagger
        self._device = device
        self._tagger = None
        self._tag_filter = None

    @property
    def tagger(self):
        """Lazy-load the RAM++ tagger."""
        if self._tagger is None:
            from perception.tagger import ImageTagger

            self._tagger = ImageTagger(device=self._device)
        return self._tagger

    @property
    def tag_filter(self):
        """Lazy-load the Claude tag filter."""
        if self._tag_filter is None:
            from perception.tag_filter import TagFilter

            self._tag_filter = TagFilter()
        return self._tag_filter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: PIL.Image.Image | str) -> PIL.Image.Image:
        """Accept a PIL Image or a file-system path and return an RGB PIL Image."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = PIL.Image.open(path)
        return image.convert("RGB")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: PIL.Image.Image | str,
        text_prompts: list[str] | None = None,
    ) -> PerceptionResult:
        """Execute the full pipeline: detect -> segment -> depth.

        Parameters
        ----------
        image:
            An RGB PIL image **or** a path to an image file.
        text_prompts:
            Optional category names to guide detection.

        Returns
        -------
        PerceptionResult
            Detections, masks, metric depth map, and image size.
        """
        image = self._load_image(image)
        image_size = image.size  # (width, height)

        # --- 1. Detection --------------------------------------------------
        logger.info("Running object detection …")

        if text_prompts is not None:
            # User provided explicit prompts — use original all-at-once DINO
            logger.info("Using explicit prompts: %s", text_prompts)
            detections = self.detector.detect(image, text_prompts=text_prompts)
        elif self.use_tagger:
            # RAM++ → Claude filter → per-tag DINO pipeline
            logger.info("Running RAM++ tagging …")
            raw_tags = self.tagger.tag(image)

            logger.info("Running Claude tag filter …")
            filtered_tags = self.tag_filter.filter_tags(raw_tags)

            # Guarantee spatial anchors are always sent to DINO
            anchors_added = []
            for anchor in sorted(SPATIAL_ANCHORS):
                if anchor not in filtered_tags:
                    filtered_tags.append(anchor)
                    anchors_added.append(anchor)
            if anchors_added:
                logger.info("Injected spatial anchors: %s", anchors_added)

            if not filtered_tags:
                logger.warning("Tag filter returned empty list — falling back to defaults.")
                detections = self.detector.detect(image, text_prompts=None)
            else:
                logger.info("Running per-tag detection for %d tags …", len(filtered_tags))
                detections = self.detector.detect_per_tag(image, filtered_tags)
        else:
            detections = self.detector.detect(image, text_prompts=None)

        if not detections:
            logger.warning("No objects detected — returning empty perception result.")
            # Still compute depth since downstream modules may need it
            depth_map = self.depth_estimator.estimate(image)
            return PerceptionResult(
                detections=[],
                masks=[],
                depth_map=depth_map,
                image_size=image_size,
            )

        # --- 2. Segmentation -----------------------------------------------
        logger.info("Running segmentation for %d detections …", len(detections))
        masks = self.segmentor.segment(image, detections)

        # --- 3. Depth -------------------------------------------------------
        logger.info("Running depth estimation …")
        depth_map = self.depth_estimator.estimate(image)

        result = PerceptionResult(
            detections=detections,
            masks=masks,
            depth_map=depth_map,
            image_size=image_size,
        )
        logger.info(
            "Perception complete: %d objects, depth shape %s, image %s",
            result.num_objects,
            depth_map.shape,
            image_size,
        )
        return result
