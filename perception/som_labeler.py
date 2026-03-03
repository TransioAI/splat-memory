"""Set-of-Mark labeling: overlay numbered markers on masks, label with Gemini."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

from perception.detector import Detection

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)

# Gemini prompt for labeling numbered segments
_LABEL_PROMPT = """\
You are looking at an image with colored segmentation overlays and numbered markers.
Each segment is shown as a semi-transparent colored region with a contour outline.
A numbered circle marker is placed at the center of each segment.
The colored region shows EXACTLY which pixels belong to that segment.

For EACH numbered segment, identify what real-world physical object that colored
region corresponds to. Pay attention to the SHAPE and EXTENT of the colored overlay,
not just what is near the number marker.

Rules:
- Use concrete, specific noun labels (e.g. "kitchen counter", "TV", "couch")
- If a segment covers a room structure, label it (e.g. "wall", "floor", "ceiling")
- If you cannot confidently identify a segment, use "unknown"
- Confidence should be 0.0-1.0 based on how certain you are of the label

Return ONLY a JSON array, no other text. Example:
[{"id": 1, "label": "couch", "confidence": 0.95}, {"id": 2, "label": "wall", "confidence": 0.8}]
"""


class SoMLabeler:
    """Label image segments using Set-of-Mark prompting with Gemini.

    Workflow:
        1. ``draw_marks()`` — overlay numbered markers on mask centroids
        2. ``label_segments()`` — send marked image to Gemini for identification
        3. ``labels_to_detections()`` — convert labels + masks to Detection objects

    The Gemini client is lazy-loaded on first use.  Requires the
    ``GEMINI_API_KEY`` environment variable.
    """

    MODEL_ID = "gemini-2.5-flash"

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self) -> None:
        """Load the Gemini client if not already loaded."""
        if self._client is not None:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for SoM labeling. "
                "Get one at https://aistudio.google.com/apikey"
            )

        from google import genai

        self._client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized (model=%s)", self.MODEL_ID)

    # ------------------------------------------------------------------
    # 1. Draw numbered markers
    # ------------------------------------------------------------------

    @staticmethod
    def draw_marks(
        image: PIL.Image.Image,
        masks: list[np.ndarray],
        scores: list[float],
    ) -> tuple[PIL.Image.Image, list[tuple[int, int]]]:
        """Overlay colored mask regions with contours and numbered markers.

        Each segment gets:
        1. A semi-transparent colored overlay so Gemini sees segment extent
        2. A contour outline for clear boundaries
        3. A numbered circle marker at the centroid

        Parameters
        ----------
        image:
            The original RGB PIL image.
        masks:
            List of ``(H, W)`` boolean mask arrays.
        scores:
            Corresponding mask scores (used only for ordering context).

        Returns
        -------
        tuple[PIL.Image.Image, list[tuple[int, int]]]
            ``(marked_image, centroids)`` where centroids is a list of
            ``(cx, cy)`` pixel coordinates for each marker.
        """
        from PIL import Image

        canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR).astype(np.float32)
        h_img, w_img = canvas.shape[:2]

        # Adaptive sizing based on image resolution
        radius = max(16, min(w_img, h_img) // 40)
        font_scale = max(0.5, radius / 18)
        thickness = max(1, radius // 7)
        contour_thickness = max(2, min(w_img, h_img) // 400)

        # Distinct color palette (BGR) — 20 colors to reduce repeats
        colors = [
            (66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 4),
            (171, 71, 188), (0, 172, 193), (255, 112, 67), (92, 107, 192),
            (38, 166, 154), (255, 167, 38), (141, 110, 99), (2, 136, 209),
            (255, 64, 129), (0, 200, 83), (170, 0, 255), (255, 214, 0),
            (0, 131, 143), (213, 0, 0), (100, 181, 246), (174, 213, 129),
        ]

        centroids: list[tuple[int, int]] = []

        # Pass 1: Draw semi-transparent colored overlays + contours
        for i, mask in enumerate(masks):
            bool_mask = mask.astype(bool)
            color = colors[i % len(colors)]

            # Semi-transparent overlay (30% opacity)
            overlay = np.zeros_like(canvas)
            overlay[bool_mask] = color
            canvas[bool_mask] = canvas[bool_mask] * 0.7 + overlay[bool_mask] * 0.3

            # Draw contour outline
            mask_u8 = bool_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas.astype(np.uint8), contours, -1, color, contour_thickness)

        canvas = canvas.astype(np.uint8)

        # Pass 2: Draw numbered markers at centroids (on top of overlays)
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                centroids.append((0, 0))
                continue

            cx = int(np.median(xs))
            cy = int(np.median(ys))
            centroids.append((cx, cy))

            color = colors[i % len(colors)]
            marker_id = i + 1  # 1-indexed

            # Black outline circle for contrast, then colored fill
            cv2.circle(canvas, (cx, cy), radius + 2, (0, 0, 0), -1)
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 2)

            # Draw number centered in circle
            label = str(marker_id)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
            )
            text_x = cx - tw // 2
            text_y = cy + th // 2
            # Black shadow for readability
            cv2.putText(
                canvas, label, (text_x + 1, text_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA,
            )

        marked_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        marked_image = Image.fromarray(marked_rgb)

        logger.info("Drew %d segment overlays + numbered markers on image", len(centroids))
        return marked_image, centroids

    # ------------------------------------------------------------------
    # 2. Label segments with Gemini
    # ------------------------------------------------------------------

    def label_segments(
        self,
        marked_image: PIL.Image.Image,
        num_segments: int,
    ) -> list[dict]:
        """Send the marked image to Gemini and get labels for each segment.

        Parameters
        ----------
        marked_image:
            The image with numbered markers overlaid.
        num_segments:
            Number of segments (markers) in the image.

        Returns
        -------
        list[dict]
            List of ``{"id": int, "label": str, "confidence": float}`` dicts,
            one per identified segment.
        """
        self._ensure_client()

        prompt = (
            f"There are {num_segments} numbered segments in this image. "
            f"{_LABEL_PROMPT}"
        )

        logger.info("Sending marked image to Gemini (%d segments) …", num_segments)

        response = self._client.models.generate_content(
            model=self.MODEL_ID,
            contents=[prompt, marked_image],
        )

        # Parse JSON from response
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            labels = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini response as JSON: %s", text[:500])
            # Return empty labels — caller can handle gracefully
            labels = []

        if not isinstance(labels, list):
            logger.error("Gemini returned non-list response: %s", type(labels))
            labels = []

        logger.info("Gemini returned %d segment labels", len(labels))
        return labels

    # ------------------------------------------------------------------
    # 3. Convert labels + masks to Detection objects
    # ------------------------------------------------------------------

    @staticmethod
    def labels_to_detections(
        labels: list[dict],
        masks: list[np.ndarray],
    ) -> tuple[list[Detection], list[int]]:
        """Convert Gemini labels + masks into Detection objects.

        Filters out segments labeled ``"unknown"`` and computes bounding boxes
        from mask extents.

        Parameters
        ----------
        labels:
            Output of ``label_segments()``.
        masks:
            The original masks from ``AutoMaskGenerator.generate()``.

        Returns
        -------
        tuple[list[Detection], list[int]]
            ``(detections, kept_mask_indices)`` — the Detection objects and the
            indices into the original masks list that were kept.
        """
        detections: list[Detection] = []
        kept_indices: list[int] = []

        for entry in labels:
            seg_id = entry.get("id")
            label = entry.get("label", "unknown")
            confidence = float(entry.get("confidence", 0.5))

            if seg_id is None:
                continue

            # Convert 1-indexed segment ID to 0-indexed mask index
            mask_idx = seg_id - 1
            if mask_idx < 0 or mask_idx >= len(masks):
                logger.warning("Segment ID %d out of range (have %d masks)", seg_id, len(masks))
                continue

            # Skip unknowns
            if label.lower() in ("unknown", "unidentified", "unclear"):
                continue

            # Compute bbox from mask extents
            mask = masks[mask_idx]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue

            x_min = float(np.min(xs))
            y_min = float(np.min(ys))
            x_max = float(np.max(xs))
            y_max = float(np.max(ys))

            detections.append(
                Detection(
                    bbox=[x_min, y_min, x_max, y_max],
                    label=label,
                    confidence=confidence,
                )
            )
            kept_indices.append(mask_idx)

        logger.info(
            "SoM labels → %d detections (%d filtered as unknown/invalid)",
            len(detections),
            len(labels) - len(detections),
        )
        return detections, kept_indices
