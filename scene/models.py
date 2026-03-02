"""Pydantic models for scene graph representation."""

from pydantic import BaseModel


class SceneObject(BaseModel):
    """A detected object with 3D spatial properties."""

    label: str
    confidence: float
    position_m: tuple[float, float, float]  # (x, y, z) in camera frame
    dimensions_m: tuple[float, float, float]  # (width, height, depth)
    distance_m: float  # distance from camera


class SceneRelation(BaseModel):
    """A spatial relation between two objects."""

    subject: str
    predicate: str
    object: str
    distance_m: float | None = None


class CalibrationInfo(BaseModel):
    """Calibration metadata."""

    reference_object: str | None = None
    scale_factor: float = 1.0
    fov_degrees: float = 70.0


class SceneGraph(BaseModel):
    """Complete scene graph with objects, relations, and calibration."""

    objects: list[SceneObject]
    relations: list[SceneRelation]
    calibration: CalibrationInfo
