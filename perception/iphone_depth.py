"""Extract metric depth from iPhone LiDAR data embedded in HEIC files.

iPhone Pro models (with LiDAR) embed a disparity map as an auxiliary image
in HEIC files.  This module extracts and converts it to metric depth.

The LiDAR sensor has a max range of ~5m.  The embedded depth image is a
low-resolution (typically 768x576) 8-bit disparity map where:
- 255 = closest to camera (~0m)
- 0   = furthest (~5m or out of range)

Conversion: ``depth_meters = (255 - disparity) / 255 * MAX_RANGE``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)

# iPhone LiDAR max range in meters
LIDAR_MAX_RANGE_M = 5.0


def extract_iphone_depth(image_path: str) -> np.ndarray | None:
    """Extract metric depth map from an iPhone HEIC file with LiDAR data.

    Parameters
    ----------
    image_path:
        Path to a HEIC file (must be from an iPhone Pro with LiDAR).

    Returns
    -------
    np.ndarray | None
        A float32 depth map in metres with shape matching the primary image,
        or None if no LiDAR depth data is found.
    """
    try:
        import pillow_heif
    except ImportError:
        logger.debug("pillow-heif not installed, cannot extract iPhone depth.")
        return None

    try:
        heif_file = pillow_heif.open_heif(image_path)
    except Exception:
        logger.debug("Failed to open %s as HEIF.", image_path)
        return None

    primary = heif_file[0]
    depth_images = primary.info.get("depth_images", [])

    if not depth_images:
        logger.debug("No depth images found in %s.", image_path)
        return None

    depth_heif = depth_images[0]
    disp = np.asarray(depth_heif)

    # Handle multi-channel depth (take first channel)
    if disp.ndim == 3:
        disp = disp[:, :, 0]

    disp = disp.astype(np.float32)

    # Convert disparity (0-255) to metric depth (0-5m)
    # 255 = closest, 0 = furthest
    depth_m = (255.0 - disp) / 255.0 * LIDAR_MAX_RANGE_M

    logger.info(
        "Extracted iPhone LiDAR depth: %dx%d, range=[%.2f, %.2f]m",
        depth_m.shape[1],
        depth_m.shape[0],
        depth_m.min(),
        depth_m.max(),
    )

    return depth_m


def resize_iphone_depth(
    depth_map: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize iPhone depth map to match the primary image dimensions.

    Parameters
    ----------
    depth_map:
        Low-resolution depth map from ``extract_iphone_depth()``.
    target_width:
        Target width (primary image width).
    target_height:
        Target height (primary image height).

    Returns
    -------
    np.ndarray
        Resized depth map of shape ``(target_height, target_width)``.
    """
    if depth_map.shape[:2] == (target_height, target_width):
        return depth_map

    resized = cv2.resize(
        depth_map,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    logger.info(
        "Resized iPhone depth from %s to (%d, %d).",
        depth_map.shape[:2],
        target_height,
        target_width,
    )
    return resized
