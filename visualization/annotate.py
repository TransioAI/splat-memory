"""OpenCV-based 2D image annotation with bboxes, labels, dimensions, and distances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from perception.detector import Detection
    from scene.models import SceneGraph

logger = logging.getLogger(__name__)

# Distinct color palette for up to 20 objects (BGR for OpenCV)
COLORS = [
    (244, 133, 66),   # blue
    (53, 67, 234),    # red
    (4, 188, 251),    # yellow
    (83, 168, 52),    # green
    (0, 109, 255),    # orange
    (188, 71, 171),   # purple
    (212, 188, 0),    # cyan
    (38, 167, 255),   # amber
    (72, 85, 121),    # brown
    (139, 125, 96),   # blue-grey
    (147, 224, 255),  # light amber
    (180, 105, 255),  # hot pink
    (50, 205, 50),    # lime green
    (255, 144, 30),   # dodger blue
    (128, 0, 128),    # purple (dark)
    (0, 215, 255),    # gold
    (203, 192, 255),  # pink
    (60, 20, 220),    # crimson
    (130, 200, 0),    # spring green
    (180, 130, 70),   # steel blue
]

# Minimum / maximum font scale bounds
_MIN_FONT_SCALE = 0.35
_MAX_FONT_SCALE = 1.8
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _font_scale(image_width: int) -> float:
    """Compute a font scale factor proportional to image width.

    Targets a readable size for images ranging from ~640px to ~4000px.
    """
    scale = image_width / 1600.0
    return float(np.clip(scale, _MIN_FONT_SCALE, _MAX_FONT_SCALE))


def _thickness(image_width: int) -> int:
    """Line / text thickness that scales with image size."""
    return max(1, round(image_width / 800.0))


def _bbox_thickness(image_width: int) -> int:
    """Bounding-box line thickness — always at least 2px."""
    return max(2, round(image_width / 600.0))


def _draw_text_with_bg(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
    fg_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    padding: int = 4,
) -> int:
    """Draw *text* with a filled background rectangle for readability.

    Parameters
    ----------
    canvas:
        Image array (mutated in-place).
    text:
        The string to render.
    origin:
        (x, y) of the top-left corner of the background rectangle.
    font_scale, thickness:
        OpenCV font rendering parameters.
    fg_color, bg_color:
        Foreground (text) and background (rect) colors in BGR.
    padding:
        Pixels of padding around the text inside the background rect.

    Returns
    -------
    int
        The height consumed (background rect height) so the caller can stack
        multiple lines vertically.
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = origin
    rect_h = th + baseline + 2 * padding

    # Clamp to image bounds
    h_img, w_img = canvas.shape[:2]
    x = max(0, min(x, w_img - tw - 2 * padding))
    y = max(0, min(y, h_img - rect_h))

    cv2.rectangle(
        canvas,
        (x, y),
        (x + tw + 2 * padding, y + rect_h),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(
        canvas,
        text,
        (x + padding, y + padding + th),
        _FONT,
        font_scale,
        fg_color,
        thickness,
        cv2.LINE_AA,
    )
    return rect_h


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Derive an axis-aligned bounding box from a binary mask.

    Returns (x_min, y_min, x_max, y_max) or None if the mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    y_min = int(row_indices[0])
    y_max = int(row_indices[-1])
    x_min = int(col_indices[0])
    x_max = int(col_indices[-1])
    return (x_min, y_min, x_max, y_max)


def _resolve_bboxes(
    num_objects: int,
    detections: list[Detection] | None,
    masks: list[np.ndarray] | None,
) -> list[tuple[int, int, int, int] | None]:
    """Return a list of bboxes (one per object), using detections first, masks as fallback.

    If neither source provides a bbox for a given index, the entry is ``None``.
    """
    bboxes: list[tuple[int, int, int, int] | None] = [None] * num_objects

    # Primary: detections carry pixel bboxes
    if detections is not None:
        for i, det in enumerate(detections):
            if i >= num_objects:
                break
            x1, y1, x2, y2 = det.bbox
            bboxes[i] = (round(x1), round(y1), round(x2), round(y2))

    # Fallback: derive from masks where detection bbox is missing
    if masks is not None:
        for i, mask in enumerate(masks):
            if i >= num_objects:
                break
            if bboxes[i] is None:
                bboxes[i] = _bbox_from_mask(mask)

    return bboxes


def annotate_image(
    image: Image.Image | np.ndarray,
    scene_graph: SceneGraph,
    detections: list[Detection] | None = None,
    masks: list[np.ndarray] | None = None,
    show_dimensions: bool = True,
    show_distance: bool = True,
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw bboxes, labels, dimensions, and distances on the image.

    Optionally overlay semi-transparent masks.

    Parameters
    ----------
    image:
        Input image as a PIL Image (RGB) or NumPy array (RGB or BGR).
    scene_graph:
        The complete scene graph containing objects and calibration info.
    detections:
        Optional list of ``Detection`` objects carrying pixel-space bboxes
        (aligned 1-to-1 with ``scene_graph.objects``).  If not provided,
        bboxes are derived from *masks* when available.
    masks:
        Optional list of boolean / uint8 masks, one per object.
    show_dimensions:
        If True, annotate each object with its estimated 3D dimensions.
    show_distance:
        If True, annotate each object with its estimated distance.
    alpha:
        Opacity of the mask overlay (0 = fully transparent, 1 = opaque).

    Returns
    -------
    np.ndarray
        Annotated image in BGR format (OpenCV convention).
    """
    # ------------------------------------------------------------------
    # 1. Convert input to BGR numpy array
    # ------------------------------------------------------------------
    if isinstance(image, Image.Image):
        canvas = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    elif isinstance(image, np.ndarray):
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Assume input might be RGB; we need BGR for OpenCV drawing.
        # Heuristic: caller typically passes RGB from PIL pipeline.
        # We treat it as BGR here since the final output is BGR anyway.
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h_img, w_img = canvas.shape[:2]
    objects = scene_graph.objects
    num_objects = len(objects)

    if num_objects == 0:
        logger.info("Scene graph has no objects; returning unannotated image.")
        return canvas

    # ------------------------------------------------------------------
    # 2. Scaling parameters
    # ------------------------------------------------------------------
    fs = _font_scale(w_img)
    thick = _thickness(w_img)
    bbox_thick = _bbox_thickness(w_img)
    pad = max(2, round(fs * 6))

    # ------------------------------------------------------------------
    # 3. Resolve bounding boxes
    # ------------------------------------------------------------------
    bboxes = _resolve_bboxes(num_objects, detections, masks)

    # ------------------------------------------------------------------
    # 4. Overlay masks (painted onto a copy, then alpha-blended)
    # ------------------------------------------------------------------
    if masks is not None and len(masks) > 0:
        overlay = canvas.copy()
        for i, mask in enumerate(masks):
            if i >= num_objects:
                break
            color = COLORS[i % len(COLORS)]
            # Ensure mask matches image dimensions
            if mask.shape[:2] != (h_img, w_img):
                logger.warning(
                    "Mask %d shape %s != image shape (%d, %d); resizing.",
                    i,
                    mask.shape[:2],
                    h_img,
                    w_img,
                )
                mask = cv2.resize(
                    mask.astype(np.uint8), (w_img, h_img), interpolation=cv2.INTER_NEAREST
                )
            bool_mask = mask.astype(bool)
            overlay[bool_mask] = color
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

    # ------------------------------------------------------------------
    # 5. Draw per-object annotations
    # ------------------------------------------------------------------
    for i, obj in enumerate(objects):
        color = COLORS[i % len(COLORS)]
        bbox = bboxes[i]

        # --- Bounding box ---
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, bbox_thick)
            text_x = x1
            text_y = max(0, y1 - pad)
        else:
            # No bbox available — place label in top-left area stacked by index
            text_x = pad
            text_y = pad + i * int(fs * 50)

        # --- Label + confidence ---
        label_text = f"{obj.label} ({obj.confidence:.0%})"
        line_h = _draw_text_with_bg(
            canvas,
            label_text,
            (text_x, text_y),
            fs,
            thick,
            fg_color=(255, 255, 255),
            bg_color=color,
        )
        current_y = text_y + line_h + 2

        # --- Dimensions ---
        if show_dimensions:
            w, h, d = obj.dimensions_m
            dim_text = f"{w:.2f} x {h:.2f} x {d:.2f}m"
            line_h = _draw_text_with_bg(
                canvas, dim_text, (text_x, current_y), fs * 0.75, thick
            )
            current_y += line_h + 2

        # --- Distance ---
        if show_distance:
            dist_text = f"{obj.distance_m:.1f}m away"
            _draw_text_with_bg(
                canvas, dist_text, (text_x, current_y), fs * 0.75, thick
            )

    # ------------------------------------------------------------------
    # 6. Info box — calibration metadata in top-right corner
    # ------------------------------------------------------------------
    _draw_calibration_info(canvas, scene_graph, fs, thick)

    logger.info("Annotated image with %d objects (%dx%d).", num_objects, w_img, h_img)
    return canvas


def _draw_calibration_info(
    canvas: np.ndarray,
    scene_graph: SceneGraph,
    fs: float,
    thick: int,
) -> None:
    """Render a small info box in the top-right corner with calibration data."""
    cal = scene_graph.calibration
    lines = [
        f"FOV: {cal.fov_degrees:.0f} deg",
        f"Scale: {cal.scale_factor:.2f}x",
    ]
    if cal.reference_object:
        lines.append(f"Ref: {cal.reference_object}")

    _h_img, w_img = canvas.shape[:2]
    info_fs = fs * 0.6
    info_thick = max(1, thick)
    padding = max(4, round(fs * 4))

    # Measure total box size
    line_heights: list[int] = []
    max_tw = 0
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, _FONT, info_fs, info_thick)
        line_heights.append(th + baseline)
        max_tw = max(max_tw, tw)

    box_w = max_tw + 2 * padding
    box_h = sum(line_heights) + (len(lines) + 1) * padding

    # Position: top-right corner with margin
    margin = max(6, round(fs * 8))
    box_x = w_img - box_w - margin
    box_y = margin

    # Semi-transparent dark background
    roi = canvas[box_y : box_y + box_h, box_x : box_x + box_w]
    overlay = roi.copy()
    bg = np.zeros_like(overlay)
    bg[:] = (40, 40, 40)
    cv2.addWeighted(bg, 0.7, overlay, 0.3, 0, dst=roi)

    # Draw border
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (200, 200, 200), 1)

    # Draw text lines
    cursor_y = box_y + padding
    for _i, line in enumerate(lines):
        (tw, th), baseline = cv2.getTextSize(line, _FONT, info_fs, info_thick)
        cursor_y += th
        cv2.putText(
            canvas,
            line,
            (box_x + padding, cursor_y),
            _FONT,
            info_fs,
            (220, 220, 220),
            info_thick,
            cv2.LINE_AA,
        )
        cursor_y += baseline + padding


def save_annotated(image: np.ndarray, output_path: str) -> None:
    """Save an annotated BGR image to disk.

    Parameters
    ----------
    image:
        Annotated image in BGR format (as returned by :func:`annotate_image`).
    output_path:
        Destination file path (e.g. ``"output/annotated.jpg"``).
    """
    success = cv2.imwrite(output_path, image)
    if success:
        logger.info("Saved annotated image to %s", output_path)
    else:
        logger.error("Failed to save annotated image to %s", output_path)
