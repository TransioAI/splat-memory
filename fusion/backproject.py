"""Back-project masked pixels to 3D using pinhole camera model or MASt3R pointmaps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MIN_VALID_POINTS = 10


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Object3D:
    """A detected object back-projected into 3D coordinates (camera or world frame)."""

    label: str
    confidence: float
    centroid: np.ndarray  # (3,) — [x, y, z] in meters
    dimensions_m: tuple[float, float, float]  # (width, height, depth) via 5th/95th percentiles
    distance_m: float  # distance from camera origin or world origin
    points_3d: np.ndarray  # (N, 3) all back-projected points


def backproject_to_3d(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    label: str,
    confidence: float,
) -> Object3D | None:
    """Back-project an object's masked pixels to 3D points using pinhole geometry.

    Args:
        mask: Boolean mask of shape (H, W) indicating object pixels.
        depth_map: Metric depth map of shape (H, W) in meters.
        intrinsics: Camera intrinsic parameters.
        label: Detected object label.
        confidence: Detection confidence score.

    Returns:
        Object3D with 3D centroid, dimensions, and point cloud, or None if
        too few valid depth pixels exist within the mask.
    """
    # Get pixel coordinates where mask is True — (v, u) = (row, col)
    vs, us = np.where(mask)

    if len(vs) == 0:
        logger.debug("Empty mask for '%s', skipping back-projection.", label)
        return None

    # Sample depth at masked pixel locations
    z_values = depth_map[vs, us]

    # Filter out zero / invalid depth values
    valid = z_values > 0
    vs = vs[valid]
    us = us[valid]
    z_values = z_values[valid]

    if len(z_values) < MIN_VALID_POINTS:
        logger.debug(
            "Only %d valid depth pixels for '%s' (need %d), skipping.",
            len(z_values),
            label,
            MIN_VALID_POINTS,
        )
        return None

    # Pinhole back-projection: image coords → 3D camera frame
    x_values = (us - intrinsics.cx) * z_values / intrinsics.fx
    y_values = (vs - intrinsics.cy) * z_values / intrinsics.fy

    points_3d = np.stack([x_values, y_values, z_values], axis=-1)  # (N, 3)

    # Centroid via median (resistant to outliers from mask edges)
    centroid = np.array([
        np.median(x_values),
        np.median(y_values),
        np.median(z_values),
    ])

    # Dimensions via 5th/95th percentiles (robust to noise)
    width = float(np.percentile(x_values, 95) - np.percentile(x_values, 5))
    height = float(np.percentile(y_values, 95) - np.percentile(y_values, 5))
    depth = float(np.percentile(z_values, 95) - np.percentile(z_values, 5))
    dimensions_m = (width, height, depth)

    distance_m = float(np.linalg.norm(centroid))

    logger.info(
        "Back-projected '%s': centroid=(%.2f, %.2f, %.2f)m, dist=%.2fm, "
        "dims=(%.2f, %.2f, %.2f)m, %d pts",
        label,
        *centroid,
        distance_m,
        *dimensions_m,
        len(points_3d),
    )

    return Object3D(
        label=label,
        confidence=confidence,
        centroid=centroid,
        dimensions_m=dimensions_m,
        distance_m=distance_m,
        points_3d=points_3d,
    )


def backproject_from_pointmap(
    mask: np.ndarray,
    pointmap_world: np.ndarray,
    confidence_map: np.ndarray,
    label: str,
    confidence: float,
    min_confidence: float = 0.5,
) -> Object3D | None:
    """Construct Object3D from MASt3R per-pixel world-coordinate pointmap.

    Unlike backproject_to_3d() which uses pinhole model + depth map,
    this takes pre-computed 3D world coordinates directly from MASt3R's
    globally aligned output. No camera intrinsics needed.

    Parameters
    ----------
    mask:
        Boolean mask (H, W) indicating object pixels.
    pointmap_world:
        (H, W, 3) array of world-frame 3D coordinates from MASt3R.
    confidence_map:
        (H, W) MASt3R confidence values per pixel.
    label:
        Object label.
    confidence:
        Detection confidence score.
    min_confidence:
        Minimum MASt3R confidence to include a point.

    Returns
    -------
    Object3D | None
        3D object in world frame, or None if too few valid points.
    """
    vs, us = np.where(mask)
    if len(vs) == 0:
        logger.debug("Empty mask for '%s', skipping pointmap back-projection.", label)
        return None

    # Get world-frame 3D points at masked pixels
    points_3d = pointmap_world[vs, us]  # (N, 3)
    conf_values = confidence_map[vs, us]  # (N,)

    # Filter by MASt3R confidence
    valid = conf_values >= min_confidence
    points_3d = points_3d[valid]

    if len(points_3d) < MIN_VALID_POINTS:
        logger.debug(
            "Only %d confident points for '%s' (need %d), skipping.",
            len(points_3d), label, MIN_VALID_POINTS,
        )
        return None

    # Centroid via median (same as single-view pipeline)
    centroid = np.median(points_3d, axis=0)

    # Dimensions via 5th/95th percentiles
    width = float(np.percentile(points_3d[:, 0], 95) - np.percentile(points_3d[:, 0], 5))
    height = float(np.percentile(points_3d[:, 1], 95) - np.percentile(points_3d[:, 1], 5))
    depth = float(np.percentile(points_3d[:, 2], 95) - np.percentile(points_3d[:, 2], 5))

    distance_m = float(np.linalg.norm(centroid))

    logger.info(
        "Pointmap back-projected '%s': centroid=(%.2f, %.2f, %.2f)m, "
        "dims=(%.2f, %.2f, %.2f)m, %d pts",
        label, *centroid, *[width, height, depth], len(points_3d),
    )

    return Object3D(
        label=label,
        confidence=confidence,
        centroid=centroid,
        dimensions_m=(width, height, depth),
        distance_m=distance_m,
        points_3d=points_3d,
    )
