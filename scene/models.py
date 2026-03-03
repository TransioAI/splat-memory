"""Pydantic models for scene graph representation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SceneObject(BaseModel):
    """A detected object with 2D bbox and 3D spatial properties."""

    id: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(
        description="[x_min, y_min, x_max, y_max] in pixels",
    )
    position_m: tuple[float, float, float] = Field(
        description="(x, y, z) centroid in meters",
    )
    dimensions_m: tuple[float, float, float] = Field(
        description="(width, height, depth) in meters",
    )
    distance_m: float = Field(description="Distance from camera/origin in meters")
    # Multi-view optional fields
    view_count: int | None = Field(
        default=None,
        description="Number of views this object was seen in (multi-view only)",
    )
    coordinate_frame: str | None = Field(
        default=None,
        description="'camera' for single-image, 'world' for multi-view",
    )


class SceneRelation(BaseModel):
    """A spatial relation between two objects."""

    subject_id: int
    subject_label: str
    predicate: str
    object_id: int
    object_label: str
    distance_m: float | None = None


class CalibrationInfo(BaseModel):
    """Calibration metadata."""

    fov_degrees: float = 70.0
    intrinsics_source: str = "default"  # "user_provided", "exif", "mast3r", or "default"
    scale_factor: float = 1.0
    reference_object: str | None = None
    image_width: int = 0
    image_height: int = 0
    # Multi-view optional fields
    coordinate_frame: str | None = Field(
        default=None,
        description="'camera' for single-image, 'world' for multi-view",
    )
    num_frames: int | None = Field(
        default=None,
        description="Number of keyframes processed (multi-view only)",
    )
    camera_poses: list[dict] | None = Field(
        default=None,
        description="Camera poses per frame (multi-view only)",
    )
    scale_warning: str | None = Field(
        default=None,
        description="Warning when no reference object found for scale calibration",
    )


class SceneGraph(BaseModel):
    """Complete scene graph with objects, relations, and calibration."""

    objects: list[SceneObject]
    relations: list[SceneRelation]
    calibration: CalibrationInfo

    def to_prompt_text(self) -> str:
        """Format scene graph as readable text for LLM context."""
        lines: list[str] = []

        # Header
        lines.append("=== SCENE GRAPH ===")
        lines.append("")

        # Calibration info
        cal = self.calibration
        is_world = cal.coordinate_frame == "world"

        if is_world:
            lines.append(
                f"Mode: Multi-view reconstruction ({cal.num_frames} keyframes)"
            )
            lines.append(
                "Coordinate frame: WORLD (positions are absolute, not camera-relative)"
            )
        else:
            lines.append(f"Image: {cal.image_width}x{cal.image_height} px")
            lines.append(
                f"FOV: {cal.fov_degrees:.1f} degrees (source: {cal.intrinsics_source})"
            )

        lines.append(f"Scale factor: {cal.scale_factor:.3f}")
        if cal.reference_object:
            lines.append(f"Reference object: {cal.reference_object}")
        if cal.scale_warning:
            lines.append(f"WARNING: {cal.scale_warning}")
        lines.append("")

        # Objects table
        lines.append("--- OBJECTS ---")
        has_views = any(obj.view_count is not None for obj in self.objects)
        header = (
            f"{'ID':<4} {'Label':<20} {'Conf':<6} "
            f"{'Position (x,y,z) m':<28} {'Dimensions (w,h,d) m':<28} {'Dist m':<8}"
        )
        if has_views:
            header += f"{'Views':<6}"
        lines.append(header)

        for obj in self.objects:
            px, py, pz = obj.position_m
            dw, dh, dd = obj.dimensions_m
            row = (
                f"{obj.id:<4} {obj.label:<20} {obj.confidence:<6.2f} "
                f"({px:+7.2f}, {py:+7.2f}, {pz:+7.2f})      "
                f"({dw:5.2f}, {dh:5.2f}, {dd:5.2f})          "
                f"{obj.distance_m:<8.2f}"
            )
            if has_views:
                row += f"{obj.view_count or 1:<6}"
            lines.append(row)
        lines.append("")

        # Relations
        if self.relations:
            lines.append("--- SPATIAL RELATIONS ---")
            for rel in self.relations:
                dist_str = (
                    f" ({rel.distance_m:.2f}m apart)"
                    if rel.distance_m is not None
                    else ""
                )
                lines.append(
                    f"{rel.subject_label} (id={rel.subject_id}) "
                    f"is {rel.predicate} "
                    f"{rel.object_label} (id={rel.object_id})"
                    f"{dist_str}"
                )
            lines.append("")

        lines.append("=== END SCENE GRAPH ===")
        return "\n".join(lines)
