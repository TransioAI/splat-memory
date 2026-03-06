"""Post-merge consolidation of structural surfaces using plane fitting.

Walls, floors, and ceilings are planar surfaces that get over-segmented
by per-frame detection + centroid-based merging. This module clusters
fragments by their fitted plane parameters (normal + offset) and merges
fragments that lie on the same geometric plane.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from video.models import MergedObject

logger = logging.getLogger(__name__)

STRUCTURAL_LABELS = {"wall", "floor", "ceiling"}

# Plane clustering thresholds
NORMAL_ANGLE_THRESHOLD_DEG = 15.0  # max angle between normals to be same plane
PLANE_OFFSET_THRESHOLD_M = 0.5     # max distance between parallel planes
MIN_POINTS_FOR_PLANE = 50          # need enough points for reliable RANSAC


def ransac_fit_plane(
    points: np.ndarray,
    n_iterations: int = 200,
    distance_threshold: float = 0.05,
) -> tuple[np.ndarray, float, float] | None:
    """Fit a plane to 3D points using RANSAC.

    Parameters
    ----------
    points:
        (N, 3) array of 3D points.
    n_iterations:
        Number of RANSAC iterations.
    distance_threshold:
        Max distance from plane to count as inlier (meters).

    Returns
    -------
    (normal, offset, inlier_ratio) or None if fitting fails.
        normal: (3,) unit normal vector
        offset: signed distance from origin
        inlier_ratio: fraction of points within distance_threshold
    """
    if len(points) < 3:
        return None

    best_normal = None
    best_offset = 0.0
    best_inliers = 0

    rng = np.random.default_rng(seed=42)
    n = len(points)

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]

        # Compute plane normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm

        # Plane offset: n . p = d
        offset = float(np.dot(normal, p0))

        # Count inliers
        distances = np.abs(np.dot(points, normal) - offset)
        inliers = int(np.sum(distances < distance_threshold))

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_offset = offset

    if best_normal is None:
        return None

    # Ensure consistent normal direction (point towards positive y = "up" for floors,
    # or use the dominant direction)
    inlier_ratio = best_inliers / n
    return best_normal, best_offset, inlier_ratio


def _normals_similar(n1: np.ndarray, n2: np.ndarray, angle_thresh_deg: float) -> bool:
    """Check if two normals point in the same direction (or opposite)."""
    cos_angle = abs(float(np.dot(n1, n2)))
    cos_angle = min(cos_angle, 1.0)  # clamp for numerical safety
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg < angle_thresh_deg


def _planes_coplanar(
    n1: np.ndarray, d1: float,
    n2: np.ndarray, d2: float,
    angle_thresh_deg: float,
    offset_thresh_m: float,
) -> bool:
    """Check if two planes are approximately the same plane."""
    if not _normals_similar(n1, n2, angle_thresh_deg):
        return False

    # For parallel planes, check offset distance
    # Handle flipped normals: if n1 ≈ -n2, flip one
    if np.dot(n1, n2) < 0:
        d2 = -d2

    return abs(d1 - d2) < offset_thresh_m


def merge_structural_surfaces(
    objects: list[MergedObject],
    angle_threshold_deg: float = NORMAL_ANGLE_THRESHOLD_DEG,
    offset_threshold_m: float = PLANE_OFFSET_THRESHOLD_M,
) -> list[MergedObject]:
    """Consolidate over-segmented structural surfaces using plane fitting.

    Parameters
    ----------
    objects:
        List of merged objects (output of cross-view merge).
    angle_threshold_deg:
        Max angle between normals to consider same plane.
    offset_threshold_m:
        Max offset distance between parallel planes.

    Returns
    -------
    list[MergedObject]
        Objects with structural surfaces consolidated.
    """
    # Separate structural from non-structural
    structural = []
    non_structural = []
    for obj in objects:
        if obj.label.lower() in STRUCTURAL_LABELS:
            structural.append(obj)
        else:
            non_structural.append(obj)

    if not structural:
        return objects

    logger.info(
        "Structural merge: %d structural objects (%s)",
        len(structural),
        ", ".join(f"{sum(1 for o in structural if o.label.lower() == l)} {l}"
                  for l in sorted(STRUCTURAL_LABELS) if any(o.label.lower() == l for o in structural)),
    )

    # Group by label first
    by_label: dict[str, list[MergedObject]] = {}
    for obj in structural:
        key = obj.label.lower()
        by_label.setdefault(key, []).append(obj)

    merged_structural = []

    for label, fragments in by_label.items():
        # Fit planes to each fragment
        plane_fits: list[tuple[MergedObject, np.ndarray, float, float]] = []

        for frag in fragments:
            pts = frag.points_3d_world
            if len(pts) < MIN_POINTS_FOR_PLANE:
                # Too few points — keep as is but mark low confidence
                merged_structural.append(frag)
                continue

            result = ransac_fit_plane(pts)
            if result is None:
                merged_structural.append(frag)
                continue

            normal, offset, inlier_ratio = result
            if inlier_ratio < 0.5:
                # Poor plane fit — this fragment isn't actually planar
                merged_structural.append(frag)
                continue

            plane_fits.append((frag, normal, offset, inlier_ratio))

        if not plane_fits:
            continue

        # Cluster fragments by plane parameters
        clusters: list[list[int]] = []
        assigned = set()

        for i, (_, n_i, d_i, _) in enumerate(plane_fits):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            for j, (_, n_j, d_j, _) in enumerate(plane_fits):
                if j in assigned:
                    continue
                if _planes_coplanar(n_i, d_i, n_j, d_j, angle_threshold_deg, offset_threshold_m):
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        # Merge each cluster into a single object
        for cluster_indices in clusters:
            cluster_frags = [plane_fits[i][0] for i in cluster_indices]
            merged_obj = _merge_fragments(cluster_frags, label)
            merged_structural.append(merged_obj)

        logger.info(
            "  %s: %d fragments with planes → %d clusters (+ %d non-planar kept)",
            label,
            len(plane_fits),
            len(clusters),
            len(fragments) - len(plane_fits),
        )

    # Re-assign IDs
    all_objects = non_structural + merged_structural
    for i, obj in enumerate(all_objects):
        obj.object_id = i

    logger.info(
        "Structural merge complete: %d objects → %d objects",
        len(objects),
        len(all_objects),
    )
    return all_objects


def _merge_fragments(fragments: list[MergedObject], label: str) -> MergedObject:
    """Merge multiple fragments into a single object."""
    # Combine all points
    all_points = np.concatenate([f.points_3d_world for f in fragments], axis=0)

    # Subsample if too many points
    max_pts = 100_000
    if len(all_points) > max_pts:
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(all_points), size=max_pts, replace=False)
        all_points = all_points[indices]

    # Centroid from all points
    centroid = all_points.mean(axis=0)

    # Dimensions from P5/P95
    dimensions = tuple(
        float(np.percentile(all_points[:, i], 95) - np.percentile(all_points[:, i], 5))
        for i in range(3)
    )

    # Max confidence across fragments
    confidence = max(f.confidence for f in fragments)

    # Total view count
    view_count = sum(f.view_count for f in fragments)

    # Average descriptor
    descriptors = np.array([f.descriptor for f in fragments])
    descriptor = descriptors.mean(axis=0)

    # Collect all frame detections
    all_frame_dets = []
    for f in fragments:
        all_frame_dets.extend(f.frame_detections)

    return MergedObject(
        object_id=0,  # will be reassigned
        label=label,
        confidence=confidence,
        centroid_world=centroid,
        dimensions_m=dimensions,
        points_3d_world=all_points,
        descriptor=descriptor,
        view_count=view_count,
        frame_detections=all_frame_dets,
    )
