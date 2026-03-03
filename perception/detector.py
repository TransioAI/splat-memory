"""Object detection using Grounding DINO (fallback: Florence-2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import PIL.Image
import torch

logger = logging.getLogger(__name__)

DEFAULT_TEXT_PROMPT = (
    "person. chair. table. door. wall. floor. couch. bed. desk. "
    "monitor. lamp. window. shelf. cabinet. countertop. appliance."
)

# Per-tag confidence threshold overrides.  Tags not listed here use the
# detector's default ``confidence_threshold`` (typically 0.3).
TAG_CONFIDENCE_OVERRIDES: dict[str, float] = {
    "door": 0.5,
    "doorway": 0.5,
    "countertop": 0.5,
    "counter": 0.5,
}


@dataclass
class Detection:
    """A single detected object."""

    bbox: list[float]  # [x_min, y_min, x_max, y_max] in pixels
    label: str
    confidence: float


class ObjectDetector:
    """Detect objects in an image using Grounding DINO or Florence-2.

    Models are lazy-loaded on the first call to ``detect()``.
    Falls back to Florence-2 when Grounding DINO cannot be loaded.
    """

    GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
    FLORENCE2_MODEL = "microsoft/Florence-2-large"

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None
        self._backend: str | None = None  # "grounding_dino" or "florence2"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_grounding_dino(self) -> bool:
        """Attempt to load Grounding DINO. Returns True on success."""
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("Loading Grounding DINO from %s …", self.GROUNDING_DINO_MODEL)
            self._processor = AutoProcessor.from_pretrained(self.GROUNDING_DINO_MODEL)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.GROUNDING_DINO_MODEL
            ).to(self.device)
            self._model.eval()
            self._backend = "grounding_dino"
            logger.info("Grounding DINO loaded successfully on %s", self.device)
            return True
        except Exception:
            logger.warning("Failed to load Grounding DINO, will try Florence-2 fallback.")
            return False

    def _load_florence2(self) -> bool:
        """Attempt to load Florence-2 as a fallback detector. Returns True on success."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info("Loading Florence-2 from %s …", self.FLORENCE2_MODEL)
            self._processor = AutoProcessor.from_pretrained(
                self.FLORENCE2_MODEL, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.FLORENCE2_MODEL, trust_remote_code=True
            ).to(self.device)
            self._model.eval()
            self._backend = "florence2"
            logger.info("Florence-2 loaded successfully on %s", self.device)
            return True
        except Exception:
            logger.error("Failed to load Florence-2 fallback.")
            return False

    def _ensure_model(self) -> None:
        """Load a detection model if one has not been loaded yet."""
        if self._model is not None:
            return
        if not self._load_grounding_dino() and not self._load_florence2():
            raise RuntimeError(
                "Could not load any detection model "
                "(tried Grounding DINO and Florence-2)."
            )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _detect_grounding_dino(
        self, image: PIL.Image.Image, text_prompt: str
    ) -> list[Detection]:
        """Run Grounding DINO inference."""
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        detections: list[Detection] = []
        for box, label, score in zip(
            results["boxes"], results["labels"], results["scores"], strict=False
        ):
            detections.append(
                Detection(
                    bbox=box.tolist() if torch.is_tensor(box) else list(box),
                    label=label.strip().rstrip("."),
                    confidence=float(score),
                )
            )
        return detections


    def _detect_grounding_dino_batch(
        self,
        image: PIL.Image.Image,
        text_prompts: list[str],
        threshold: float,
    ) -> list[list[Detection]]:
        """Run batched Grounding DINO inference — one forward pass for N prompts."""
        batch_size = len(text_prompts)
        images = [image] * batch_size
        target_sizes = [image.size[::-1]] * batch_size  # (height, width)

        inputs = self._processor(
            images=images, text=text_prompts, return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=target_sizes,
        )

        batch_detections: list[list[Detection]] = []
        for result in results:
            detections: list[Detection] = []
            for box, label, score in zip(
                result["boxes"], result["labels"], result["scores"], strict=False
            ):
                detections.append(
                    Detection(
                        bbox=box.tolist() if torch.is_tensor(box) else list(box),
                        label=label.strip().rstrip("."),
                        confidence=float(score),
                    )
                )
            batch_detections.append(detections)

        return batch_detections

    def _detect_florence2(
        self, image: PIL.Image.Image, text_prompt: str
    ) -> list[Detection]:
        """Run Florence-2 object detection via the ``<OD>`` task.

        Florence-2 uses a generative approach.  The ``<OD>`` task returns
        bounding boxes and labels without requiring a text prompt.  When a
        text prompt is supplied we use ``<CAPTION_TO_PHRASE_GROUNDING>``
        instead so that the results are filtered to the requested categories.
        """
        # Decide which Florence-2 task to use
        if text_prompt and text_prompt.strip():
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            prompt_text = task + text_prompt
        else:
            task = "<OD>"
            prompt_text = task

        inputs = self._processor(text=prompt_text, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task, image_size=image.size
        )

        # Florence-2 returns different keys depending on the task
        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])

        detections: list[Detection] = []
        for bbox, label in zip(bboxes, labels, strict=False):
            detections.append(
                Detection(
                    bbox=list(bbox),
                    label=label.strip().rstrip("."),
                    confidence=1.0,  # Florence-2 does not output confidence scores
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_per_tag(
        self,
        image: PIL.Image.Image,
        tags: list[str],
        iou_threshold: float = 0.5,
        return_pre_nms: bool = False,
    ) -> list[Detection] | tuple[list[Detection], list[Detection]]:
        """Detect objects by running DINO once per tag, then merging with NMS.

        This produces more accurate detections than sending all tags at once,
        at the cost of N forward passes (one per tag).

        Parameters
        ----------
        image:
            An RGB PIL image.
        tags:
            List of individual object labels (e.g. ``["couch", "table", "lamp"]``).
        iou_threshold:
            IoU threshold for cross-category NMS.
        return_pre_nms:
            When True, return ``(post_nms, pre_nms)`` instead of just post-NMS.

        Returns
        -------
        list[Detection] | tuple[list[Detection], list[Detection]]
            NMS-filtered detections, or ``(post_nms, pre_nms)`` if *return_pre_nms*.
        """
        self._ensure_model()
        image = image.convert("RGB")

        all_detections: list[Detection] = []
        default_threshold = self.confidence_threshold
        for tag in tags:
            tag_clean = tag.strip().rstrip(".")
            text_prompt = tag_clean + "."

            # Apply per-tag threshold override if configured
            override = TAG_CONFIDENCE_OVERRIDES.get(tag_clean.lower())
            if override is not None:
                self.confidence_threshold = override
            else:
                self.confidence_threshold = default_threshold

            logger.debug(
                "Running detection for tag: '%s' (threshold=%.2f)",
                tag_clean, self.confidence_threshold,
            )

            if self._backend == "grounding_dino":
                dets = self._detect_grounding_dino(image, text_prompt)
            else:
                dets = self._detect_florence2(image, text_prompt)

            logger.debug("  Tag '%s': %d detections", tag_clean, len(dets))
            all_detections.extend(dets)

        # Restore default threshold
        self.confidence_threshold = default_threshold

        logger.info(
            "Per-tag detection: %d tags → %d raw detections",
            len(tags),
            len(all_detections),
        )

        from perception.nms import cross_category_nms

        filtered = cross_category_nms(all_detections, iou_threshold=iou_threshold)

        logger.info(
            "After NMS: %d detections (backend=%s)",
            len(filtered),
            self._backend,
        )

        if return_pre_nms:
            return filtered, all_detections
        return filtered

    def detect(
        self,
        image: PIL.Image.Image,
        text_prompts: list[str] | None = None,
    ) -> list[Detection]:
        """Detect objects in *image*.

        Parameters
        ----------
        image:
            An RGB PIL image.
        text_prompts:
            Optional list of category names.  When *None* a broad set of
            indoor-scene categories is used.

        Returns
        -------
        list[Detection]
            Bounding boxes, labels, and confidence scores.
        """
        self._ensure_model()

        # Build a single period-separated text prompt from the list
        if text_prompts is not None:
            text_prompt = ". ".join(p.strip().rstrip(".") for p in text_prompts) + "."
        else:
            text_prompt = DEFAULT_TEXT_PROMPT

        image = image.convert("RGB")

        if self._backend == "grounding_dino":
            detections = self._detect_grounding_dino(image, text_prompt)
        else:
            detections = self._detect_florence2(image, text_prompt)

        logger.info(
            "Detected %d objects (backend=%s, threshold=%.2f)",
            len(detections),
            self._backend,
            self.confidence_threshold,
        )
        return detections
