"""Top-down 2D floorplan from multi-view reconstruction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from fusion.backproject import Object3D

logger = logging.getLogger(__name__)

# Reuse the same color palette as pointcloud.py
_PLOTLY_COLORS = [
    "rgb(66,133,244)",
    "rgb(234,67,53)",
    "rgb(251,188,4)",
    "rgb(52,168,83)",
    "rgb(255,109,0)",
    "rgb(171,71,188)",
    "rgb(0,188,212)",
    "rgb(255,167,38)",
    "rgb(121,85,72)",
    "rgb(96,125,139)",
]


def render_floorplan(
    objects_3d: list[Object3D],
    camera_poses: list | None = None,
    max_points_per_object: int = 2000,
    margin_m: float = 0.5,
    output_html: str | None = None,
) -> go.Figure:
    """Render a top-down floorplan by projecting onto the XZ plane.

    - Objects shown as scatter points sized by their XZ footprint
    - Object labels at centroid positions
    - Camera trajectory shown as dotted path
    - Grid at 1m intervals
    - Y axis (height) is collapsed; only X (right) and Z (depth) shown

    Parameters
    ----------
    objects_3d:
        List of Object3D instances in world frame.
    camera_poses:
        Optional list of CameraPose instances.
    max_points_per_object:
        Max points per object for XZ scatter.
    margin_m:
        Margin around the scene extent in meters.
    output_html:
        Optional file path for standalone HTML output.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if not objects_3d:
        fig.update_layout(title="Top-Down Floorplan (no objects)")
        if output_html:
            fig.write_html(output_html)
        return fig

    # Project each object's points onto XZ plane
    for i, obj in enumerate(objects_3d):
        color = _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        pts = obj.points_3d

        if pts is None or len(pts) == 0:
            continue

        # Subsample
        if len(pts) > max_points_per_object:
            rng = np.random.default_rng(seed=i)
            indices = rng.choice(len(pts), size=max_points_per_object, replace=False)
            pts = pts[indices]

        # XZ projection (X = right, Z = depth)
        fig.add_trace(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.3),
            name=obj.label,
            hoverinfo="skip",
        ))

        # Centroid label
        cx, _, cz = obj.centroid
        fig.add_trace(go.Scatter(
            x=[float(cx)],
            y=[float(cz)],
            mode="markers+text",
            marker=dict(size=10, color=color, symbol="diamond"),
            text=[obj.label],
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"{obj.label} (centroid)",
            showlegend=False,
            hovertemplate=(
                f"<b>{obj.label}</b><br>"
                f"X: {cx:.2f}m, Z: {cz:.2f}m<br>"
                f"Width: {obj.dimensions_m[0]:.2f}m, Depth: {obj.dimensions_m[2]:.2f}m"
                "<extra></extra>"
            ),
        ))

    # Camera trajectory
    if camera_poses:
        positions = np.array([
            cp.position_world if hasattr(cp, "position_world") else cp["position"]
            for cp in camera_poses
        ])

        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 2],
            mode="lines+markers",
            marker=dict(
                size=5,
                color=list(range(len(positions))),
                colorscale="Bluered",
                showscale=False,
            ),
            line=dict(color="gray", width=1, dash="dot"),
            name="Camera path",
            hovertemplate=(
                "Frame %{text}<br>"
                "X: %{x:.2f}m, Z: %{y:.2f}m"
                "<extra></extra>"
            ),
            text=[
                str(cp.frame_idx if hasattr(cp, "frame_idx") else i)
                for i, cp in enumerate(camera_poses)
            ],
        ))

    # Layout with equal aspect ratio and 1m grid
    fig.update_layout(
        xaxis=dict(
            title="X [m]",
            dtick=1.0,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Z [m]",
            scaleanchor="x",
            dtick=1.0,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        title="Top-Down Floorplan (XZ Projection)",
        legend=dict(itemsizing="constant", font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=50),
    )

    if output_html:
        fig.write_html(output_html)
        logger.info("Saved floorplan to %s", output_html)

    logger.info("Rendered floorplan with %d objects.", len(objects_3d))
    return fig
