"""Cross-view object merging using 3D bounding-box IoU, spatial proximity, and labels."""

from __future__ import annotations

import logging

import numpy as np

from video.models import FrameDetection, MergedObject

logger = logging.getLogger(__name__)

# Merge thresholds
BBOX3D_IOU_THRESHOLD = 0.05  # min 3D bbox IoU for merge (low because partial views)
SPATIAL_PROXIMITY_M = 1.5  # fallback: max centroid distance when IoU is 0 but objects are close
FEATURE_COSINE_THRESHOLD = 0.5  # relaxed: appearance changes across views
MAX_POINTS_PER_OBJECT = 50_000  # cap accumulated points to limit memory


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def bbox3d_from_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned 3D bounding box from points.

    Returns (min_corner, max_corner), each shape (3,).
    """
    if len(pts) == 0:
        return np.zeros(3), np.zeros(3)
    return pts.min(axis=0), pts.max(axis=0)


def bbox3d_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute IoU of axis-aligned 3D bounding boxes from two point sets.

    Uses P5/P95 percentiles instead of min/max to be robust to outliers.
    """
    if len(pts_a) < 3 or len(pts_b) < 3:
        return 0.0

    # Use percentiles for robustness
    min_a = np.percentile(pts_a, 5, axis=0)
    max_a = np.percentile(pts_a, 95, axis=0)
    min_b = np.percentile(pts_b, 5, axis=0)
    max_b = np.percentile(pts_b, 95, axis=0)

    # Intersection
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))

    # Volumes
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))

    union = vol_a + vol_b - inter_vol
    if union < 1e-12:
        return 0.0
    return inter_vol / union


def extract_mask_descriptor(
    mask: np.ndarray,
    descriptor_map: np.ndarray,
) -> np.ndarray:
    """Average MASt3R descriptors over masked pixels."""
    vs, us = np.where(mask)
    if len(vs) == 0:
        return np.zeros(descriptor_map.shape[-1], dtype=np.float32)
    descs = descriptor_map[vs, us]
    return descs.mean(axis=0).astype(np.float32)


def merge_objects_across_views(
    frame_detections: list[list[FrameDetection]],
    iou_threshold: float = BBOX3D_IOU_THRESHOLD,
    spatial_threshold: float = SPATIAL_PROXIMITY_M,
    feature_threshold: float = FEATURE_COSINE_THRESHOLD,
    require_label_match: bool = True,
) -> list[MergedObject]:
    """Merge detections across frames into globally unique objects.

    Uses a two-tier matching strategy:
    - Primary: 3D bounding-box IoU (view-independent, robust to partial segmentation)
    - Fallback: centroid proximity for small or thin objects where IoU is unreliable

    Feature cosine similarity is checked but with a relaxed threshold (0.5)
    since appearance changes significantly across viewpoints.

    Parameters
    ----------
    frame_detections:
        List of per-frame detection lists.
    iou_threshold:
        Minimum 3D bbox IoU for merge.
    spatial_threshold:
        Fallback max centroid distance when IoU is zero.
    feature_threshold:
        Minimum cosine similarity (relaxed for cross-view).
    require_label_match:
        Whether labels must match exactly.
    """
    merged: list[MergedObject] = []
    next_id = 0

    for frame_dets in frame_detections:
        for det in frame_dets:
            best_match: MergedObject | None = None
            best_score = -float("inf")
            has_descriptors = np.linalg.norm(det.descriptor) > 1e-8

            for obj in merged:
                # Criterion 1: Label match
                if require_label_match and det.label.lower() != obj.label.lower():
                    continue

                # Criterion 2: 3D bounding-box IoU
                iou = bbox3d_iou(det.points_3d_world, obj.points_3d_world)

                # Fallback: centroid distance for small/thin objects
                centroid_dist = float(np.linalg.norm(det.centroid_world - obj.centroid_world))

                if iou < iou_threshold and centroid_dist > spatial_threshold:
                    continue

                # Criterion 3: Feature similarity (relaxed)
                if has_descriptors and np.linalg.norm(obj.descriptor) > 1e-8:
                    feat_sim = cosine_similarity(det.descriptor, obj.descriptor)
                    if feat_sim < feature_threshold:
                        continue
                else:
                    feat_sim = 1.0

                # Score: IoU dominates, centroid proximity as tiebreaker
                score = iou * 10.0 + feat_sim / (1.0 + centroid_dist)
                if score > best_score:
                    best_match = obj
                    best_score = score

            if best_match is not None:
                _update_merged(best_match, det)
            else:
                merged.append(MergedObject(
                    object_id=next_id,
                    label=det.label,
                    confidence=det.confidence,
                    centroid_world=det.centroid_world.copy(),
                    dimensions_m=det.dimensions_m,
                    points_3d_world=det.points_3d_world.copy(),
                    descriptor=det.descriptor.copy(),
                    view_count=1,
                    frame_detections=[det],
                ))
                next_id += 1

    logger.info(
        "Merged %d frame detections into %d unique objects.",
        sum(len(fd) for fd in frame_detections),
        len(merged),
    )
    return merged


def _update_merged(obj: MergedObject, det: FrameDetection) -> None:
    """Update a merged object with a new detection (running average)."""
    n = obj.view_count

    # Running average centroid
    obj.centroid_world = (obj.centroid_world * n + det.centroid_world) / (n + 1)

    # Running average descriptor
    obj.descriptor = (obj.descriptor * n + det.descriptor) / (n + 1)

    # Max confidence
    obj.confidence = max(obj.confidence, det.confidence)

    # Accumulate points (subsample to limit memory)
    combined = np.concatenate([obj.points_3d_world, det.points_3d_world], axis=0)
    if len(combined) > MAX_POINTS_PER_OBJECT:
        rng = np.random.default_rng(seed=obj.object_id)
        indices = rng.choice(len(combined), size=MAX_POINTS_PER_OBJECT, replace=False)
        combined = combined[indices]
    obj.points_3d_world = combined

    # Refine dimensions from accumulated points (P5/P95)
    obj.dimensions_m = (
        float(np.percentile(combined[:, 0], 95) - np.percentile(combined[:, 0], 5)),
        float(np.percentile(combined[:, 1], 95) - np.percentile(combined[:, 1], 5)),
        float(np.percentile(combined[:, 2], 95) - np.percentile(combined[:, 2], 5)),
    )

    obj.view_count = n + 1
    obj.frame_detections.append(det)
