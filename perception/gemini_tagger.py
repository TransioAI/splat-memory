"""Image tagging using Gemini 2.5 Flash — replacement for RAM++ + Claude filter."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)

_TAG_PROMPT = """\
List every distinct physical object visible in this image.

Include:
- Furniture: chairs, tables, desks, couches, shelves, beds, cabinets, etc.
- Appliances: refrigerator, microwave, oven, washing machine, etc.
- Electronics: monitor, TV, laptop, phone, speaker, etc.
- Structural elements: wall, floor, ceiling, door, window, countertop, staircase
- Containers: box, bin, basket, bag, suitcase
- Decorative: plant, picture frame, clock, lamp, vase, rug
- Any other tangible, individually identifiable items

Rules:
- Use lowercase noun phrases (e.g. "office chair", "cardboard box", "kitchen counter")
- Each tag should be specific enough for an object detector to draw a bounding box around it
- Do NOT include: materials (wood, metal), colors alone (white, brown), \
scene labels (living room, indoor), abstract concepts (modern, cozy), \
actions (sitting, standing), or broad categories (furniture, electronics)
- Deduplicate synonyms — pick the most common term (couch not sofa, monitor not screen)
- If multiple instances of the same object exist, list the tag only ONCE

Return ONLY a JSON array of strings, no other text.
Example: ["office chair", "desk", "monitor", "keyboard", "wall", "floor", "window", "plant"]
"""


class GeminiTagger:
    """Tag an image with object labels using Gemini 2.5 Flash.

    Replaces the RAM++ tagger + Claude tag filter with a single Gemini call.
    The model is lazy-loaded on first use.  Requires the ``GEMINI_API_KEY``
    environment variable.

    Parameters
    ----------
    model_id:
        Gemini model to use.
    """

    MODEL_ID = "gemini-2.5-flash"

    def __init__(self, model_id: str | None = None) -> None:
        self._client = None
        if model_id:
            self.MODEL_ID = model_id

    def _ensure_client(self) -> None:
        """Load the Gemini client if not already loaded."""
        if self._client is not None:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for Gemini tagger. "
                "Get one at https://aistudio.google.com/apikey"
            )

        from google import genai

        self._client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized (model=%s)", self.MODEL_ID)

    def tag(self, image: PIL.Image.Image) -> list[str]:
        """Tag an image and return a list of object labels.

        Parameters
        ----------
        image:
            An RGB PIL image.

        Returns
        -------
        list[str]
            Object tags suitable as individual Grounding DINO prompts.
            Already filtered — no need for a separate tag filter step.
        """
        self._ensure_client()

        logger.info("Sending image to Gemini for tagging …")

        response = self._client.models.generate_content(
            model=self.MODEL_ID,
            contents=[_TAG_PROMPT, image],
        )

        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            tags = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini response as JSON: %s", text[:500])
            return []

        if not isinstance(tags, list):
            logger.error("Gemini returned non-list response: %s", type(tags))
            return []

        tags = [str(t).strip().lower() for t in tags if str(t).strip()]

        logger.info("Gemini tagger produced %d tags: %s", len(tags), tags)
        return tags
