"""Back-project masked pixels to 3D using pinhole camera model."""

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
    """A detected object back-projected into 3D camera-frame coordinates."""

    label: str
    confidence: float
    centroid: np.ndarray  # (3,) — [x, y, z] in meters
    dimensions_m: tuple[float, float, float]  # (width, height, depth) via 5th/95th percentiles
    distance_m: float  # distance from camera origin
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
