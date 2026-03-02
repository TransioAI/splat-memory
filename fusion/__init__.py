"""Fusion module: 2D detections + metric depth → 3D scene understanding."""

from .backproject import CameraIntrinsics, Object3D, backproject_to_3d
from .calibration import apply_scale, auto_calibrate_scale, estimate_intrinsics
from .spatial_relations import SpatialRelation, compute_spatial_relations

__all__ = [
    "CameraIntrinsics",
    "Object3D",
    "SpatialRelation",
    "apply_scale",
    "auto_calibrate_scale",
    "backproject_to_3d",
    "compute_spatial_relations",
    "estimate_intrinsics",
]
