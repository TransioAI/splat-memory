"""Data models for multi-view video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraPose:
    """Camera extrinsic parameters in world frame."""

    frame_idx: int
    cam2world: np.ndarray  # (4, 4) camera-to-world transformation matrix
    focal_length: float  # focal length in pixels (from MASt3R)
    principal_point: tuple[float, float]  # (cx, cy) in pixels
    image_path: str  # path to the keyframe image

    @property
    def position_world(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return self.cam2world[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix (world frame)."""
        return self.cam2world[:3, :3]


@dataclass
class FrameDetection:
    """A detection from a single keyframe, with world-frame 3D data."""

    frame_idx: int
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coords
    centroid_world: np.ndarray  # (3,) world-frame centroid
    dimensions_m: tuple[float, float, float]  # (width, height, depth) meters
    points_3d_world: np.ndarray  # (N, 3) world-frame points
    descriptor: np.ndarray  # (D,) averaged MASt3R descriptor over mask
    mask: np.ndarray  # (H, W) boolean mask


@dataclass
class MergedObject:
    """An object merged across multiple views."""

    object_id: int
    label: str
    confidence: float  # max confidence across views
    centroid_world: np.ndarray  # (3,) running average centroid
    dimensions_m: tuple[float, float, float]  # refined from accumulated points
    points_3d_world: np.ndarray  # (N, 3) accumulated world-frame points
    descriptor: np.ndarray  # (D,) running average descriptor
    view_count: int  # number of frames this object was seen in
    frame_detections: list[FrameDetection] = field(default_factory=list)

    @property
    def distance_from_origin(self) -> float:
        """Euclidean distance from world origin."""
        return float(np.linalg.norm(self.centroid_world))
