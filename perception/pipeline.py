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

    # Optional debug data (populated when tagger pipeline runs)
    raw_tags: list[str] | None = None
    filtered_tags: list[str] | None = None
    anchors_injected: list[str] | None = None
    pre_nms_detections: list[Detection] | None = None

    # SoM mode: the marked image with numbered overlays
    som_marked_image: PIL.Image.Image | None = None

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
        self._auto_mask_gen = None
        self._som_labeler = None

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

    @property
    def auto_mask_gen(self):
        """Lazy-load the SAM2 auto-mask generator."""
        if self._auto_mask_gen is None:
            from perception.auto_mask import AutoMaskGenerator

            self._auto_mask_gen = AutoMaskGenerator(device=self._device)
        return self._auto_mask_gen

    @property
    def som_labeler(self):
        """Lazy-load the SoM labeler (Gemini)."""
        if self._som_labeler is None:
            from perception.som_labeler import SoMLabeler

            self._som_labeler = SoMLabeler()
        return self._som_labeler

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
    # SoM pipeline
    # ------------------------------------------------------------------

    def _run_som(self, image: PIL.Image.Image) -> PerceptionResult:
        """Run the Set-of-Mark pipeline: auto-mask → mark → Gemini label → depth.

        Parameters
        ----------
        image:
            An RGB PIL image.

        Returns
        -------
        PerceptionResult
            Detections (from Gemini labels), masks (from SAM2 auto-mask),
            metric depth map, and image size.  The ``som_marked_image`` field
            is populated with the numbered-marker overlay.
        """
        image_size = image.size

        # 1. Auto-mask: SAM2 segments everything
        logger.info("SoM: running SAM2 auto-mask generation …")
        masks, scores = self.auto_mask_gen.generate(image)

        if not masks:
            logger.warning("SoM: no masks generated — returning empty result.")
            depth_map = self.depth_estimator.estimate(image)
            return PerceptionResult(
                detections=[], masks=[], depth_map=depth_map,
                image_size=image_size,
            )

        logger.info("SoM: %d masks after filtering", len(masks))

        # 2. Draw numbered markers on mask centroids
        marked_image, _centroids = self.som_labeler.draw_marks(image, masks, scores)

        # 3. Send marked image to Gemini for labeling
        labels = self.som_labeler.label_segments(marked_image, len(masks))

        # 4. Convert labels + masks to Detection objects
        detections, kept_indices = self.som_labeler.labels_to_detections(labels, masks)

        # Keep only the masks that correspond to kept detections
        kept_masks = [masks[i] for i in kept_indices]

        if not detections:
            logger.warning("SoM: no labeled detections — returning empty result.")
            depth_map = self.depth_estimator.estimate(image)
            return PerceptionResult(
                detections=[], masks=[], depth_map=depth_map,
                image_size=image_size,
            )

        # 5. Depth estimation
        logger.info("SoM: running depth estimation …")
        depth_map = self.depth_estimator.estimate(image)

        result = PerceptionResult(
            detections=detections,
            masks=kept_masks,
            depth_map=depth_map,
            image_size=image_size,
            som_marked_image=marked_image,
        )
        logger.info(
            "SoM perception complete: %d objects, depth shape %s",
            result.num_objects, depth_map.shape,
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: PIL.Image.Image | str,
        extra_objects: list[str] | None = None,
        use_som: bool = False,
    ) -> PerceptionResult:
        """Execute the full pipeline: detect -> segment -> depth.

        Parameters
        ----------
        image:
            An RGB PIL image **or** a path to an image file.
        extra_objects:
            Optional object categories to detect.  When the tagger is enabled,
            these are **merged** into the auto-discovered tags so that both
            automatic and user-specified objects are detected.  When the tagger
            is disabled, these are used as the sole detection targets.
        use_som:
            When True, use the Set-of-Mark pipeline (SAM2 auto-mask + Gemini
            labeling) instead of the standard tag → detect → segment flow.

        Returns
        -------
        PerceptionResult
            Detections, masks, metric depth map, and image size.
        """
        image = self._load_image(image)

        if use_som:
            return self._run_som(image)

        image_size = image.size  # (width, height)

        # --- 1. Detection --------------------------------------------------
        logger.info("Running object detection …")

        # Debug data (populated when tagger pipeline runs)
        _raw_tags = None
        _filtered_tags = None
        _anchors_injected = None
        _pre_nms_detections = None

        if self.use_tagger:
            # RAM++ → Claude filter → per-tag DINO pipeline
            logger.info("Running RAM++ tagging …")
            _raw_tags = self.tagger.tag(image)

            logger.info("Running Claude tag filter …")
            _filtered_tags = self.tag_filter.filter_tags(_raw_tags)

            # Merge user-specified objects into filtered tags
            if extra_objects:
                for obj in extra_objects:
                    if obj not in _filtered_tags:
                        _filtered_tags.append(obj)
                logger.info("Merged user-specified objects: %s", extra_objects)

            # Guarantee spatial anchors are always sent to DINO
            _anchors_injected = []
            for anchor in sorted(SPATIAL_ANCHORS):
                if anchor not in _filtered_tags:
                    _filtered_tags.append(anchor)
                    _anchors_injected.append(anchor)
            if _anchors_injected:
                logger.info("Injected spatial anchors: %s", _anchors_injected)

            if not _filtered_tags:
                logger.warning("Tag filter returned empty list — falling back to defaults.")
                detections = self.detector.detect(image, text_prompts=None)
            else:
                logger.info("Running per-tag detection for %d tags …", len(_filtered_tags))
                detections, _pre_nms_detections = self.detector.detect_per_tag(
                    image, _filtered_tags, return_pre_nms=True,
                )
        elif extra_objects:
            # No tagger — use user-specified objects as detection targets
            logger.info("Using user-specified objects: %s", extra_objects)
            detections = self.detector.detect(image, text_prompts=extra_objects)
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
            raw_tags=_raw_tags,
            filtered_tags=_filtered_tags,
            anchors_injected=_anchors_injected,
            pre_nms_detections=_pre_nms_detections,
        )
        logger.info(
            "Perception complete: %d objects, depth shape %s, image %s",
            result.num_objects,
            depth_map.shape,
            image_size,
        )
        return result
