"""Cross-view object merging using spatial proximity, features, and labels."""

from __future__ import annotations

import logging

import numpy as np

from video.models import FrameDetection, MergedObject

logger = logging.getLogger(__name__)

# Merge thresholds
SPATIAL_PROXIMITY_M = 0.5  # max centroid distance for merge candidate
FEATURE_COSINE_THRESHOLD = 0.7  # min cosine similarity for merge
MAX_POINTS_PER_OBJECT = 50_000  # cap accumulated points to limit memory


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_mask_descriptor(
    mask: np.ndarray,
    descriptor_map: np.ndarray,
) -> np.ndarray:
    """Average MASt3R descriptors over masked pixels.

    Parameters
    ----------
    mask:
        (H, W) boolean mask.
    descriptor_map:
        (H, W, D) per-pixel descriptor array from MASt3R.

    Returns
    -------
    np.ndarray
        (D,) averaged descriptor vector.
    """
    vs, us = np.where(mask)
    if len(vs) == 0:
        return np.zeros(descriptor_map.shape[-1], dtype=np.float32)

    descs = descriptor_map[vs, us]  # (N, D)
    return descs.mean(axis=0).astype(np.float32)


def merge_objects_across_views(
    frame_detections: list[list[FrameDetection]],
    spatial_threshold: float = SPATIAL_PROXIMITY_M,
    feature_threshold: float = FEATURE_COSINE_THRESHOLD,
    require_label_match: bool = True,
) -> list[MergedObject]:
    """Merge detections across frames into globally unique objects.

    For each detection, find the best matching existing merged object using
    three criteria (ALL must pass):
    1. Labels match (case-insensitive)
    2. Centroid distance < spatial_threshold
    3. Cosine similarity of descriptors > feature_threshold

    If a descriptor is all zeros (fallback mode), skip criterion 3.

    Parameters
    ----------
    frame_detections:
        List of per-frame detection lists. Outer index is frame order.
    spatial_threshold:
        Maximum centroid distance (meters) for merge candidacy.
    feature_threshold:
        Minimum cosine similarity of averaged descriptors.
    require_label_match:
        Whether labels must match exactly.

    Returns
    -------
    list[MergedObject]
        Globally unique objects with merged point clouds.
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

                # Criterion 2: Spatial proximity
                dist = float(np.linalg.norm(det.centroid_world - obj.centroid_world))
                if dist > spatial_threshold:
                    continue

                # Criterion 3: Feature similarity (skip if no descriptors)
                if has_descriptors and np.linalg.norm(obj.descriptor) > 1e-8:
                    feat_sim = cosine_similarity(det.descriptor, obj.descriptor)
                    if feat_sim < feature_threshold:
                        continue
                else:
                    feat_sim = 1.0  # skip feature check in fallback mode

                # Combined score: higher similarity + closer distance
                score = feat_sim / (1.0 + dist)
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
