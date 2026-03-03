"""IoU-based Non-Maximum Suppression across categories."""

from __future__ import annotations

import logging

from perception.detector import Detection

logger = logging.getLogger(__name__)


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def cross_category_nms(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Apply NMS across all detections regardless of category.

    When two boxes overlap above *iou_threshold*, the one with higher
    confidence is kept.  Detections are sorted by confidence descending
    and processed greedily.

    Parameters
    ----------
    detections:
        All detections from per-tag DINO runs, potentially overlapping.
    iou_threshold:
        IoU above which the lower-confidence detection is suppressed.

    Returns
    -------
    list[Detection]
        Filtered detections with overlapping duplicates removed.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep: list[Detection] = []
    for det in sorted_dets:
        suppressed = False
        for kept in keep:
            if compute_iou(det.bbox, kept.bbox) > iou_threshold:
                suppressed = True
                logger.debug(
                    "NMS: suppressed '%s' (%.2f) overlapping with '%s' (%.2f)",
                    det.label,
                    det.confidence,
                    kept.label,
                    kept.confidence,
                )
                break
        if not suppressed:
            keep.append(det)

    logger.info(
        "Cross-category NMS: %d → %d detections (iou_threshold=%.2f)",
        len(detections),
        len(keep),
        iou_threshold,
    )
    return keep
