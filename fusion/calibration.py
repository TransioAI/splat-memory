"""Camera intrinsics estimation and auto-scale calibration."""

from __future__ import annotations

import logging
import math

import numpy as np

from .backproject import CameraIntrinsics, Object3D

logger = logging.getLogger(__name__)

# Typical real-world heights (meters) for common indoor objects.
# Used as reference anchors for scale calibration.
KNOWN_SIZES: dict[str, float] = {
    "door": 2.03,
    "doorway": 2.03,
    "countertop": 0.91,
    "counter": 0.91,
    "person": 1.70,
    "human": 1.70,
    "dining table": 0.76,
    "table": 0.76,
    "chair": 0.85,
    "refrigerator": 1.70,
    "fridge": 1.70,
}


def estimate_intrinsics(
    image_width: int,
    image_height: int,
    fov_degrees: float = 70.0,
) -> CameraIntrinsics:
    """Estimate pinhole camera intrinsics from image dimensions.

    Assumes a smartphone-like camera with square pixels and the given
    horizontal field of view.

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fov_degrees: Horizontal field of view in degrees (default 70).

    Returns:
        CameraIntrinsics with estimated focal lengths and principal point.
    """
    fov_rad = math.radians(fov_degrees)
    fx = image_width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx  # square pixels assumption

    cx = image_width / 2.0
    cy = image_height / 2.0

    logger.info(
        "Estimated intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f (fov=%.0f°)",
        fx,
        fy,
        cx,
        cy,
        fov_degrees,
    )

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def _match_known_size(label: str, known_sizes: dict[str, float]) -> float | None:
    """Match a detected label against known sizes using case-insensitive partial matching.

    Returns the known height in meters, or None if no match is found.
    """
    label_lower = label.lower()
    for known_label, known_height in known_sizes.items():
        if known_label in label_lower or label_lower in known_label:
            return known_height
    return None


def auto_calibrate_scale(
    objects: list[Object3D],
    known_sizes: dict[str, float] | None = None,
) -> float:
    """Auto-calibrate depth scale using detected objects with known real-world sizes.

    Compares the estimated HEIGHT (dimensions_m index 1) of detected objects
    against known reference heights. If multiple reference objects are found,
    the scale factors are averaged.

    Args:
        objects: List of back-projected 3D objects.
        known_sizes: Override mapping of label → known height in meters.
            Defaults to KNOWN_SIZES if not provided.

    Returns:
        Scale factor to apply to all 3D measurements. Returns 1.0 if no
        reference object is found.
    """
    sizes = known_sizes if known_sizes is not None else KNOWN_SIZES
    scale_factors: list[float] = []

    for obj in objects:
        known_height = _match_known_size(obj.label, sizes)
        if known_height is None:
            continue

        estimated_height = obj.dimensions_m[1]  # height is index 1
        if estimated_height <= 0:
            logger.warning(
                "Object '%s' has non-positive estimated height %.3fm, skipping.",
                obj.label,
                estimated_height,
            )
            continue

        factor = known_height / estimated_height
        scale_factors.append(factor)
        logger.info(
            "Scale ref '%s': known=%.2fm, estimated=%.2fm, factor=%.3f",
            obj.label,
            known_height,
            estimated_height,
            factor,
        )

    if not scale_factors:
        logger.info("No known-size reference objects found, using scale_factor=1.0.")
        return 1.0

    avg_factor = float(np.mean(scale_factors))
    logger.info(
        "Auto-calibrated scale_factor=%.3f from %d reference object(s).",
        avg_factor,
        len(scale_factors),
    )
    return avg_factor


def apply_scale(objects: list[Object3D], scale_factor: float) -> list[Object3D]:
    """Apply a global scale factor to all 3D measurements.

    Scales centroids, dimensions, distances, and raw point clouds in-place
    and returns the updated list.

    Args:
        objects: List of Object3D instances to rescale.
        scale_factor: Multiplicative scale factor.

    Returns:
        The same list of objects with scaled 3D measurements.
    """
    if scale_factor == 1.0:
        return objects

    for obj in objects:
        obj.centroid = obj.centroid * scale_factor
        obj.dimensions_m = (
            obj.dimensions_m[0] * scale_factor,
            obj.dimensions_m[1] * scale_factor,
            obj.dimensions_m[2] * scale_factor,
        )
        obj.distance_m = obj.distance_m * scale_factor
        obj.points_3d = obj.points_3d * scale_factor

    logger.info("Applied scale_factor=%.3f to %d objects.", scale_factor, len(objects))
    return objects
