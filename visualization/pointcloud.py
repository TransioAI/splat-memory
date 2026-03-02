"""Interactive 3D point cloud visualization using Plotly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from fusion.backproject import Object3D

logger = logging.getLogger(__name__)

# RGB color palette matching the annotation module (but in 0-1 range for Plotly)
_PLOTLY_COLORS = [
    "rgb(66,133,244)",   # blue
    "rgb(234,67,53)",    # red
    "rgb(251,188,4)",    # yellow
    "rgb(52,168,83)",    # green
    "rgb(255,109,0)",    # orange
    "rgb(171,71,188)",   # purple
    "rgb(0,188,212)",    # cyan
    "rgb(255,167,38)",   # amber
    "rgb(121,85,72)",    # brown
    "rgb(96,125,139)",   # blue-grey
    "rgb(147,224,255)",  # light amber
    "rgb(255,105,180)",  # hot pink
    "rgb(50,205,50)",    # lime green
    "rgb(30,144,255)",   # dodger blue
    "rgb(128,0,128)",    # purple (dark)
    "rgb(255,215,0)",    # gold
    "rgb(255,192,203)",  # pink
    "rgb(220,20,60)",    # crimson
    "rgb(0,200,130)",    # spring green
    "rgb(70,130,180)",   # steel blue
]


def render_pointcloud_3d(
    objects_3d: list[Object3D],
    max_points_per_object: int = 5000,
    output_html: str | None = None,
) -> go.Figure:
    """Render an interactive 3D point cloud with per-object coloring.

    Each ``Object3D`` contributes a scatter trace for its point cloud and a
    larger labeled marker at its centroid.  Points are subsampled when they
    exceed *max_points_per_object* to keep the browser responsive.

    Parameters
    ----------
    objects_3d:
        List of ``Object3D`` instances (from ``fusion.backproject``).
    max_points_per_object:
        Maximum number of points rendered per object.  Points are randomly
        sub-sampled (without replacement) when exceeded.
    output_html:
        Optional file path.  When provided the figure is saved as a
        self-contained HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object, ready for ``fig.show()`` or embedding.
    """
    import plotly.graph_objects as go  # lazy import — plotly is optional / heavy

    fig = go.Figure()

    if not objects_3d:
        logger.info("No 3D objects to render; returning empty figure.")
        fig.update_layout(
            title="3D Scene Point Cloud (no objects)",
            scene=dict(
                xaxis_title="X (right) [m]",
                yaxis_title="Y (down) [m]",
                zaxis_title="Z (depth) [m]",
            ),
        )
        if output_html:
            fig.write_html(output_html)
        return fig

    for i, obj in enumerate(objects_3d):
        color = _PLOTLY_COLORS[i % len(_PLOTLY_COLORS)]
        pts = obj.points_3d  # (N, 3)

        if pts is None or len(pts) == 0:
            logger.debug("Object '%s' has no 3D points, skipping.", obj.label)
            continue

        # ------------------------------------------------------------------
        # Sub-sample if needed
        # ------------------------------------------------------------------
        n_pts = len(pts)
        if n_pts > max_points_per_object:
            rng = np.random.default_rng(seed=i)
            indices = rng.choice(n_pts, size=max_points_per_object, replace=False)
            pts = pts[indices]
            logger.debug(
                "Sub-sampled '%s' from %d to %d points.",
                obj.label,
                n_pts,
                max_points_per_object,
            )

        # ------------------------------------------------------------------
        # Point cloud trace
        # ------------------------------------------------------------------
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.6),
                name=obj.label,
                hovertemplate=(
                    f"<b>{obj.label}</b><br>"
                    "x: %{x:.3f}m<br>"
                    "y: %{y:.3f}m<br>"
                    "z: %{z:.3f}m"
                    "<extra></extra>"
                ),
            )
        )

        # ------------------------------------------------------------------
        # Centroid marker (larger, labeled)
        # ------------------------------------------------------------------
        cx, cy, cz = obj.centroid
        fig.add_trace(
            go.Scatter3d(
                x=[float(cx)],
                y=[float(cy)],
                z=[float(cz)],
                mode="markers+text",
                marker=dict(size=7, color=color, opacity=1.0, symbol="diamond"),
                text=[obj.label],
                textposition="top center",
                textfont=dict(size=11, color=color),
                name=f"{obj.label} (centroid)",
                showlegend=False,
                hovertemplate=(
                    f"<b>{obj.label} centroid</b><br>"
                    f"x: {cx:.3f}m<br>"
                    f"y: {cy:.3f}m<br>"
                    f"z: {cz:.3f}m<br>"
                    f"dist: {obj.distance_m:.2f}m<br>"
                    f"dims: {obj.dimensions_m[0]:.2f} x "
                    f"{obj.dimensions_m[1]:.2f} x "
                    f"{obj.dimensions_m[2]:.2f}m"
                    "<extra></extra>"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        scene=dict(
            xaxis_title="X (right) [m]",
            yaxis_title="Y (down) [m]",
            zaxis_title="Z (depth) [m]",
            aspectmode="data",
        ),
        title="3D Scene Point Cloud",
        legend=dict(
            itemsizing="constant",
            font=dict(size=12),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if output_html:
        fig.write_html(output_html)
        logger.info("Saved 3D point cloud to %s", output_html)

    logger.info(
        "Rendered 3D point cloud with %d objects (%d total traces).",
        len(objects_3d),
        len(fig.data),
    )
    return fig
