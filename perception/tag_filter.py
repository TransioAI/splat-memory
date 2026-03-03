"""Filter RAM++ tags using Claude to produce clean Grounding DINO prompts."""

from __future__ import annotations

import json
import logging
import re

import anthropic

logger = logging.getLogger(__name__)

TAG_FILTER_SYSTEM_PROMPT = """\
You are a preprocessing filter in a spatial reasoning pipeline.

PIPELINE CONTEXT:
1. RAM++ (open-vocabulary image tagging model) produced the tags below from a photo.
2. Each tag you keep will be sent as a SEPARATE query to Grounding DINO object detector.
3. Grounding DINO will draw a bounding box for each tag.
4. Those boxes feed into SAM2 for pixel-level segmentation → Depth Anything V2 for metric \
depth estimation → pinhole back-projection to 3D point clouds.
5. The final 3D scene graph (objects with positions, dimensions, spatial relations) is used \
for spatial reasoning questions like "what's to the left of the table?" or \
"how far is the chair from the wall?"

YOUR TASK:
Filter the RAM++ tags to ONLY keep tags that represent physically distinct, \
individually detectable entities in the image.

KEEP:
- Specific physical objects a detector can draw a tight bounding box around: \
chair, table, lamp, monitor, book, cup, bottle, plant, phone, etc.
- Surfaces and architectural elements — these are CRITICAL spatial anchors: \
floor, wall, ceiling, countertop, shelf, staircase, doorway.
- Doors and windows — these are physical landmarks with known real-world sizes \
(used for automatic scale calibration in the 3D reconstruction).

REMOVE:
- Hypernyms and broad categories: furniture, appliance, electronics, kitchenware, device.
- Scene and environment labels: living room, indoor, outdoor, kitchen, bedroom, office.
- Abstract concepts, styles, and aesthetics: modern, cozy, elegant, vintage, minimalist, \
comfortable, decorative.
- Materials and textures: wood, glass, metal, fabric, ceramic, marble, leather, plastic.
- Non-boxable phenomena: shadow, light, reflection, sunlight, glare, space, air.
- Actions, states, and properties: sitting, standing, hanging, open, closed, small, large.
- Colors when used alone: white, brown, gray (but keep "white board" if it means a whiteboard).

DEDUPLICATION:
- Keep only one term for synonyms, using the more common/specific name:
  couch + sofa → couch | tv + television + monitor → monitor | \
  rug + carpet → rug | curtain + drape → curtain
- "dining table" + "table" → table (unless both are visible as distinct objects)

Output ONLY a JSON array of strings. No explanation, no markdown fences, no preamble.
Example: ["chair", "table", "lamp", "floor", "wall", "window", "plant"]
"""


class TagFilter:
    """Use Claude with extended thinking to filter raw RAM++ tags.

    Follows the same Anthropic client pattern as
    :class:`reasoning.llm.SpatialReasoner`.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.client = anthropic.Anthropic()
        self.model = model

    def filter_tags(self, raw_tags: list[str]) -> list[str]:
        """Filter raw tags into clean, boxable object labels.

        Parameters
        ----------
        raw_tags:
            Raw tag list from RAM++ (e.g. ``["couch", "furniture", "table",
            "living room", "indoor", "modern", "floor", "sofa"]``).

        Returns
        -------
        list[str]
            Filtered, deduplicated tags suitable as individual Grounding DINO
            prompts (e.g. ``["couch", "table", "floor"]``).

        Raises
        ------
        RuntimeError
            If Claude's response cannot be parsed as a JSON array.
        """
        if not raw_tags:
            logger.warning("No tags to filter — returning empty list.")
            return []

        tags_str = json.dumps(raw_tags)
        user_message = f"Filter these image tags:\n{tags_str}"

        logger.info("Sending %d tags to Claude for filtering …", len(raw_tags))

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                },
                system=TAG_FILTER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            logger.error("Claude API error during tag filtering: %s", exc)
            raise

        # Extract text block from response (skip thinking blocks)
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        logger.debug("Claude tag filter raw response: %s", text_content)

        # Parse JSON array from response
        try:
            filtered = json.loads(text_content.strip())
        except json.JSONDecodeError as exc:
            # Fallback: extract JSON array from surrounding text
            match = re.search(r"\[.*\]", text_content, re.DOTALL)
            if match:
                filtered = json.loads(match.group())
            else:
                logger.error(
                    "Could not parse Claude tag filter response as JSON: %s",
                    text_content,
                )
                raise RuntimeError(
                    f"Tag filter returned unparseable response: {text_content!r}"
                ) from exc

        if not isinstance(filtered, list):
            raise RuntimeError(f"Tag filter returned {type(filtered)}, expected list")

        filtered = [str(tag).strip() for tag in filtered if str(tag).strip()]

        logger.info(
            "Tag filter: %d raw → %d filtered tags: %s",
            len(raw_tags),
            len(filtered),
            filtered,
        )
        return filtered
