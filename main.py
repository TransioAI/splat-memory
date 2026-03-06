"""FastAPI server and CLI entry point for splat-memory."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import argparse
import io
import logging
import signal
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from PIL import Image
from pydantic import BaseModel

from reasoning.llm import SpatialReasoner
from scene.models import CalibrationInfo, SceneGraph, SceneObject, SceneRelation

# Register HEIC/HEIF opener so PIL can handle iPhone native images
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("splat-memory")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Splat Memory",
    description="Single-image spatial reasoning: 3D scene understanding from a single photo",
    version="0.1.0",
)

# Serve web frontend
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    @app.get("/", include_in_schema=False)
    async def root():
        return HTMLResponse((_STATIC_DIR / "index.html").read_text())


# ---------------------------------------------------------------------------
# Debug artifacts cache
# ---------------------------------------------------------------------------


@dataclass
class DebugArtifacts:
    """In-memory cache of debug visualizations for a scene."""

    source_name: str | None = None
    raw_tags: list[str] | None = None
    filtered_tags: list[str] | None = None
    anchors_injected: list[str] | None = None
    pre_nms_detections: list[dict] | None = None
    post_nms_detections: list[dict] | None = None
    scene_graph_json: dict | None = None
    scene_graph_text: str | None = None
    # Image artifacts stored as JPEG bytes
    detections_jpg: bytes | None = None
    masks_jpg: bytes | None = None
    depth_jpg: bytes | None = None
    annotated_jpg: bytes | None = None
    # Per-frame mask overlays (video mode)
    frame_masks_jpg: list[bytes] | None = None
    # Interactive HTML
    pointcloud_html: str | None = None
    floorplan_html: str | None = None


# In-memory scene cache: scene_id -> (SceneGraph, SpatialReasoner, DebugArtifacts)
_scene_cache: dict[str, tuple[SceneGraph, SpatialReasoner, DebugArtifacts]] = {}

_CACHE_DIR = Path(__file__).parent / ".scene_cache"

def _save_scene_to_disk(scene_id: str, scene_graph: SceneGraph, debug: DebugArtifacts) -> None:
    import pickle
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / f"{scene_id}.pkl"
    try:
        with open(path, "wb") as f:
            pickle.dump({"scene_graph": scene_graph, "debug": debug}, f)
        logger.info("Saved scene %s to disk.", scene_id)
    except Exception as exc:
        logger.warning("Failed to save scene %s: %s", scene_id, exc)

def _load_scenes_from_disk() -> None:
    import pickle
    if not _CACHE_DIR.is_dir():
        return
    for path in sorted(_CACHE_DIR.glob("*.pkl")):
        scene_id = path.stem
        if scene_id in _scene_cache:
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            sg = data["scene_graph"]
            debug = data["debug"]
            reasoner = SpatialReasoner()
            reasoner.set_scene(sg)
            _scene_cache[scene_id] = (sg, reasoner, debug)
            logger.info("Restored scene %s from disk (%d objects).", scene_id, len(sg.objects))
        except Exception as exc:
            logger.warning("Failed to load scene %s: %s", scene_id, exc)

_load_scenes_from_disk()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    scene_id: str
    scene_graph: SceneGraph
    intrinsics_source: str


class AskRequest(BaseModel):
    scene_id: str | None = None
    question: str


class AskResponse(BaseModel):
    answer: str
    scene_id: str


# ---------------------------------------------------------------------------
# Lazy global perception pipeline
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline(use_tagger: bool = True):
    """Lazily initialise the perception pipeline (heavy model loading)."""
    global _pipeline
    if _pipeline is None:
        from perception.pipeline import PerceptionPipeline

        _pipeline = PerceptionPipeline(use_tagger=use_tagger)
    return _pipeline


# ---------------------------------------------------------------------------
# Debug rendering helpers (in-memory, no disk I/O)
# ---------------------------------------------------------------------------


def _encode_jpg(bgr_array: np.ndarray) -> bytes:
    """Encode a BGR numpy array as JPEG bytes."""
    success, buf = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _render_detections_jpg(image: Image.Image, detections: list) -> bytes:
    """Draw detection boxes on image and return JPEG bytes."""
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    _h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, w / 2000)
    thickness = max(1, w // 600)

    colors = [
        (66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 4),
        (171, 71, 188), (0, 172, 193), (255, 112, 67), (92, 107, 192),
        (38, 166, 154), (255, 167, 38), (141, 110, 99), (2, 136, 209),
    ]

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)

        label = f"{det.label} ({det.confidence:.0%})"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(th + 8, y1 - 8)
        cv2.rectangle(canvas, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(
            canvas, label, (x1 + 4, label_y - 4),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return _encode_jpg(canvas)


def _render_masks_jpg(image: Image.Image, detections: list, masks: list) -> bytes:
    """Overlay masks on image and return JPEG bytes."""
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR).astype(np.float32)

    colors = [
        (66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 4),
        (171, 71, 188), (0, 172, 193), (255, 112, 67), (92, 107, 192),
    ]

    for i, (det, mask) in enumerate(zip(detections, masks, strict=False)):
        color = colors[i % len(colors)]
        bool_mask = mask.astype(bool)
        overlay = np.zeros_like(canvas)
        overlay[bool_mask] = color
        canvas[bool_mask] = canvas[bool_mask] * 0.5 + overlay[bool_mask] * 0.5

        ys, xs = np.where(bool_mask)
        if len(ys) > 0:
            cx, cy = int(np.median(xs)), int(np.median(ys))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                canvas, det.label, (cx, cy),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    return _encode_jpg(canvas.astype(np.uint8))


def _render_depth_jpg(depth_map: np.ndarray) -> bytes:
    """Render depth map as colorized heatmap and return JPEG bytes."""
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return _encode_jpg(np.zeros_like(depth_map, dtype=np.uint8))

    d_min = float(np.percentile(valid, 2))
    d_max = float(np.percentile(valid, 98))
    normalized = np.clip((depth_map - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colored, f"{d_min:.1f}m", (10, 30), font, 0.7, (255, 255, 255), 2)
    h = colored.shape[0]
    cv2.putText(colored, f"{d_max:.1f}m", (10, h - 15), font, 0.7, (255, 255, 255), 2)

    return _encode_jpg(colored)


def _render_annotated_jpg(
    image: Image.Image, scene_graph: SceneGraph, detections: list, masks: list,
) -> bytes:
    """Full annotation using visualization/annotate.py, returned as JPEG bytes."""
    from visualization.annotate import annotate_image

    bgr = annotate_image(image, scene_graph, detections=detections, masks=masks)
    return _encode_jpg(bgr)


def _render_pointcloud_html(objects_3d: list) -> str:
    """Render interactive 3D point cloud and return HTML string."""
    from visualization.pointcloud import render_pointcloud_3d

    fig = render_pointcloud_3d(objects_3d)
    return fig.to_html(include_plotlyjs=True, full_html=True)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def analyze_image_full(
    image: Image.Image,
    detect: list[str] | None = None,
    use_tagger: bool = True,
    use_sam3: bool = False,
    fov_override: float | None = None,
) -> tuple[str, SceneGraph]:
    """Run the full perception + fusion pipeline and build a SceneGraph.

    Steps:
        1. Resolve FOV (user override > EXIF > default 70°).
        2. Run perception pipeline (detect, segment, depth).
        3. Estimate camera intrinsics.
        4. Back-project each detection + mask to 3D.
        5. Auto-calibrate scale using known-size reference objects.
        6. Apply scale correction.
        7. Compute pairwise spatial relations.
        8. Assemble SceneGraph.
        9. Generate debug artifacts.
        10. Cache with a UUID scene_id.

    Parameters
    ----------
    image:
        A PIL image (EXIF should still be intact — extract before .convert("RGB")).
    detect:
        Optional list of additional object categories to detect.  When the
        tagger is enabled, these are merged into the auto-discovered tags.
    use_tagger:
        When True, use RAM++ → Claude filter → per-tag DINO pipeline.
    fov_override:
        User-provided FOV in degrees. Takes priority over EXIF and default.

    Returns
    -------
    tuple[str, SceneGraph]
        ``(scene_id, scene_graph)``
    """
    from fusion.backproject import backproject_to_3d
    from fusion.calibration import (
        KNOWN_SIZES,
        apply_scale,
        auto_calibrate_scale,
        estimate_intrinsics,
        extract_fov_from_exif,
    )
    from fusion.spatial_relations import compute_spatial_relations

    # 1. Resolve FOV: user override > EXIF > default
    if fov_override is not None and fov_override > 0:
        fov_degrees = fov_override
        intrinsics_source = "user_provided"
        logger.info("Using user-provided FOV: %.1f°", fov_degrees)
    else:
        # Extract EXIF before converting to RGB (which strips metadata)
        exif_fov = extract_fov_from_exif(image)
        if exif_fov is not None:
            fov_degrees = exif_fov
            intrinsics_source = "exif"
            logger.info("Using EXIF-derived FOV: %.1f°", fov_degrees)
        else:
            fov_degrees = 70.0
            intrinsics_source = "default"
            logger.warning(
                "No EXIF metadata found — using default FOV of %.1f°. "
                "Lateral positions may be inaccurate. Pass fov_degrees or "
                "focal_length_35mm for better results.",
                fov_degrees,
            )

    # Now convert to RGB for processing
    image = image.convert("RGB")
    width, height = image.size

    pipeline = get_pipeline(use_tagger=use_tagger)

    # 2. Perception: detect -> segment -> depth
    logger.info("Running perception pipeline on %dx%d image...", width, height)

    if use_sam3:
        # SAM3: unified detection + segmentation
        from perception.sam3_detector import Sam3Detector
        from perception.gemini_tagger import GeminiTagger

        tagger = GeminiTagger()
        tags = tagger.tag(image)
        if detect:
            for obj in detect:
                if obj.lower() not in [t.lower() for t in tags]:
                    tags.append(obj)
        logger.info("SAM3 mode: %d tags from Gemini", len(tags))

        sam3 = Sam3Detector(device="cuda")
        detections, masks = sam3.detect_and_segment(image, tags)
        depth_map = pipeline.depth_estimator.estimate(image)

        # Populate result for debug artifacts
        result = type("R", (), {
            "detections": detections, "masks": masks, "depth_map": depth_map,
            "image_size": (width, height), "num_objects": len(detections),
            "raw_tags": None, "filtered_tags": tags,
            "anchors_injected": None, "pre_nms_detections": None,
        })()
    else:
        result = pipeline.run(image, extra_objects=detect)
        detections = result.detections
        masks = result.masks
        depth_map = result.depth_map

    logger.info("Perception complete: %d detections.", len(detections))

    # 3. Estimate camera intrinsics
    intrinsics = estimate_intrinsics(width, height, fov_degrees=fov_degrees)

    # 4. Back-project each detection to 3D
    objects_3d = []
    for det, mask in zip(detections, masks, strict=False):
        obj = backproject_to_3d(
            mask=mask,
            depth_map=depth_map,
            intrinsics=intrinsics,
            label=det.label,
            confidence=det.confidence,
        )
        if obj is not None:
            objects_3d.append((det, obj))

    logger.info("Back-projected %d / %d detections to 3D.", len(objects_3d), len(detections))

    # 5. Auto-calibrate scale
    all_obj3d = [obj for _, obj in objects_3d]
    scale_factor = auto_calibrate_scale(all_obj3d)

    # 6. Apply scale correction
    if scale_factor != 1.0:
        apply_scale(all_obj3d, scale_factor)

    # Find reference object label (first match from known sizes)
    reference_object: str | None = None
    for obj in all_obj3d:
        label_lower = obj.label.lower()
        if any(k in label_lower or label_lower in k for k in KNOWN_SIZES):
            reference_object = obj.label
            break

    # 7. Compute spatial relations
    spatial_rels = compute_spatial_relations(all_obj3d)

    # 8. Assemble SceneGraph
    scene_objects: list[SceneObject] = []
    for idx, (det, obj) in enumerate(objects_3d):
        scene_objects.append(
            SceneObject(
                id=idx,
                label=obj.label,
                confidence=obj.confidence,
                bbox=det.bbox,
                position_m=(
                    round(float(obj.centroid[0]), 4),
                    round(float(obj.centroid[1]), 4),
                    round(float(obj.centroid[2]), 4),
                ),
                dimensions_m=(
                    round(obj.dimensions_m[0], 4),
                    round(obj.dimensions_m[1], 4),
                    round(obj.dimensions_m[2], 4),
                ),
                distance_m=round(obj.distance_m, 4),
            )
        )

    scene_relations: list[SceneRelation] = []
    for rel in spatial_rels:
        scene_relations.append(
            SceneRelation(
                subject_id=rel.subject_idx,
                subject_label=rel.subject,
                predicate=rel.predicate,
                object_id=rel.object_idx,
                object_label=rel.object_label,
                distance_m=round(rel.distance_m, 4),
            )
        )

    calibration = CalibrationInfo(
        fov_degrees=round(fov_degrees, 2),
        intrinsics_source=intrinsics_source,
        scale_factor=round(scale_factor, 4),
        reference_object=reference_object,
        image_width=width,
        image_height=height,
    )

    scene_graph = SceneGraph(
        objects=scene_objects,
        relations=scene_relations,
        calibration=calibration,
    )

    # 9. Generate debug artifacts
    debug = DebugArtifacts(
        source_name="image",
        raw_tags=result.raw_tags,
        filtered_tags=result.filtered_tags,
        anchors_injected=result.anchors_injected,
        pre_nms_detections=[
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in result.pre_nms_detections
        ]
        if result.pre_nms_detections
        else None,
        post_nms_detections=[
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in detections
        ],
        scene_graph_json=scene_graph.model_dump(),
        scene_graph_text=scene_graph.to_prompt_text(),
    )

    # Render image-based artifacts
    try:
        debug.detections_jpg = _render_detections_jpg(image, detections)
    except Exception:
        logger.warning("Failed to render detections image.", exc_info=True)

    try:
        debug.masks_jpg = _render_masks_jpg(image, detections, masks)
    except Exception:
        logger.warning("Failed to render masks image.", exc_info=True)

    try:
        debug.depth_jpg = _render_depth_jpg(depth_map)
    except Exception:
        logger.warning("Failed to render depth image.", exc_info=True)

    try:
        debug.annotated_jpg = _render_annotated_jpg(image, scene_graph, detections, masks)
    except Exception:
        logger.warning("Failed to render annotated image.", exc_info=True)

    try:
        debug.pointcloud_html = _render_pointcloud_html(all_obj3d)
    except Exception:
        logger.warning("Failed to render point cloud.", exc_info=True)

    # 10. Cache
    scene_id = str(uuid.uuid4())
    reasoner = SpatialReasoner()
    reasoner.set_scene(scene_graph)
    _scene_cache[scene_id] = (scene_graph, reasoner, debug)
    _save_scene_to_disk(scene_id, scene_graph, debug)

    logger.info(
        "Scene %s built: %d objects, %d relations, scale=%.3f, fov=%.1f° (%s)",
        scene_id,
        len(scene_objects),
        len(scene_relations),
        scale_factor,
        fov_degrees,
        intrinsics_source,
    )

    return scene_id, scene_graph


# ---------------------------------------------------------------------------
# Video analysis function
# ---------------------------------------------------------------------------


def analyze_video_full(
    source: str | Path,
    detect: list[str] | None = None,
    use_gemini_tagger: bool = False,
    use_sam3: bool = False,
    every_n_frames: int = 30,
    max_frames: int = 40,
) -> tuple[str, SceneGraph]:
    """Run the multi-view video pipeline and build a SceneGraph.

    Parameters
    ----------
    source:
        Path to video file (.mp4/.mov) or directory of images.
    detect:
        Additional object categories to detect.
    use_gemini_tagger:
        Use Gemini 2.5 Flash for tagging instead of RAM++ + Claude.
    every_n_frames:
        Keyframe extraction interval (video only).
    max_frames:
        Maximum keyframes to process.

    Returns
    -------
    tuple[str, SceneGraph]
        ``(scene_id, scene_graph)``
    """
    from video.video_pipeline import VideoPipeline

    pipeline = VideoPipeline()
    result = pipeline.run(
        source=source,
        detect=detect,
        use_gemini_tagger=use_gemini_tagger,
        use_sam3=use_sam3,
        every_n_frames=every_n_frames,
        max_frames=max_frames,
    )

    scene_graph = result.scene_graph

    # Generate debug artifacts
    debug = DebugArtifacts(
        source_name=Path(source).name if source else "video",
        scene_graph_json=scene_graph.model_dump(),
        scene_graph_text=scene_graph.to_prompt_text(),
    )

    # Render per-frame mask overlays
    try:
        import io
        from PIL import ImageDraw, ImageFont
        frame_masks = []
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        colors = [
            (255,80,80), (80,255,80), (80,80,255), (255,255,80),
            (255,80,255), (80,255,255), (128,0,0), (0,128,0),
            (0,0,128), (128,128,0), (128,0,128), (0,128,128),
            (255,128,0), (128,255,0), (0,128,255), (255,0,128),
        ]

        for frame_idx, image in enumerate(result.keyframe_images):
            img = image.convert("RGB")
            img_array = np.array(img)

            # Get detections for this frame from pipeline
            if hasattr(result, '_all_frame_detections') and frame_idx < len(result.all_frame_detections):
                frame_dets = result.all_frame_detections[frame_idx]
            else:
                frame_dets = []

            for j, det in enumerate(frame_dets):
                color = colors[j % len(colors)]
                if det.mask is not None:
                    mask_arr = np.array(det.mask) if not isinstance(det.mask, np.ndarray) else det.mask
                    if mask_arr.ndim == 2 and mask_arr.shape[0] == img_array.shape[0]:
                        mask_bool = mask_arr > 0.5
                        for c in range(3):
                            img_array[:,:,c] = np.where(mask_bool,
                                (img_array[:,:,c] * 0.5 + color[c] * 0.5).astype(np.uint8),
                                img_array[:,:,c])

            overlay = Image.fromarray(img_array)
            draw = ImageDraw.Draw(overlay)
            for j, det in enumerate(frame_dets):
                color = colors[j % len(colors)]
                x1, y1, x2, y2 = det.bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"{det.label} {det.confidence:.2f}"
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=(0, 0, 0))
                draw.text((x1 + 3, y1 - th - 2), label, fill=(255, 255, 255), font=font)

            # Header
            header = f"Frame {frame_idx}: {len(frame_dets)} detections"
            hbbox = draw.textbbox((0, 0), header, font=font)
            hw, hh = hbbox[2] - hbbox[0], hbbox[3] - hbbox[1]
            draw.rectangle([0, 0, hw + 12, hh + 8], fill=(0, 0, 0))
            draw.text((6, 4), header, fill=(255, 255, 255), font=font)

            buf = io.BytesIO()
            overlay.save(buf, format="JPEG", quality=85)
            frame_masks.append(buf.getvalue())

        debug.frame_masks_jpg = frame_masks
        logger.info("Rendered %d per-frame mask overlays.", len(frame_masks))
    except Exception:
        logger.warning("Failed to render per-frame masks.", exc_info=True)

    # Render multi-view point cloud with camera trajectory
    try:
        from visualization.pointcloud import render_multiview_pointcloud_3d

        fig = render_multiview_pointcloud_3d(result.objects_3d, result.camera_poses)
        debug.pointcloud_html = fig.to_html(include_plotlyjs=True, full_html=True)
    except Exception:
        logger.warning("Failed to render multi-view point cloud.", exc_info=True)

    # Render top-down floorplan
    try:
        from visualization.floorplan import render_floorplan

        fig = render_floorplan(result.objects_3d, result.camera_poses)
        debug.floorplan_html = fig.to_html(include_plotlyjs=True, full_html=True)
    except Exception:
        logger.warning("Failed to render floorplan.", exc_info=True)

    # Cache
    scene_id = str(uuid.uuid4())
    reasoner = SpatialReasoner()
    reasoner.set_scene(scene_graph)
    _scene_cache[scene_id] = (scene_graph, reasoner, debug)
    _save_scene_to_disk(scene_id, scene_graph, debug)

    logger.info(
        "Video scene %s built: %d objects, %d relations, %d frames, scale=%.3f",
        scene_id,
        len(scene_graph.objects),
        len(scene_graph.relations),
        scene_graph.calibration.num_frames or 0,
        scene_graph.calibration.scale_factor,
    )

    return scene_id, scene_graph


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),  # noqa: B008
    detect: str | None = Form(None),
    fov_degrees: float | None = Form(None),
    focal_length_35mm: float | None = Form(None),
    use_sam3: bool = Form(False),
):
    """Upload an image and get back a scene graph with 3D spatial info.

    Parameters
    ----------
    file:
        Image file (JPEG, PNG, HEIC, etc.).
    detect:
        Optional comma-separated list of additional object categories to detect.
        These are merged with auto-discovered tags from the image.
    fov_degrees:
        Optional horizontal FOV override in degrees. Takes priority over EXIF.
    focal_length_35mm:
        Optional 35mm-equivalent focal length. Converted to FOV if fov_degrees
        is not provided.
    """
    # Validate content type (allow HEIC which may report as application/octet-stream)
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    detect_objects: list[str] | None = None
    if detect:
        detect_objects = [p.strip() for p in detect.split(",") if p.strip()]

    # Resolve FOV override: explicit fov_degrees > focal_length_35mm conversion
    fov_override: float | None = fov_degrees
    if fov_override is None and focal_length_35mm is not None and focal_length_35mm > 0:
        from fusion.calibration import focal_length_35mm_to_fov

        fov_override = focal_length_35mm_to_fov(focal_length_35mm)

    try:
        scene_id, scene_graph = analyze_image_full(
            image, detect=detect_objects, use_sam3=use_sam3, fov_override=fov_override,
        )
    except Exception as exc:
        logger.exception("Analysis failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(
        scene_id=scene_id,
        scene_graph=scene_graph,
        intrinsics_source=scene_graph.calibration.intrinsics_source,
    )


@app.post("/analyze_video", response_model=AnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(...),  # noqa: B008
    detect: str | None = Form(None),
    use_gemini_tagger: bool = Form(False),
    use_sam3: bool = Form(False),
    every_n_frames: int = Form(30),
    max_frames: int = Form(40),
):
    """Upload a video and get a global 3D scene graph.

    Parameters
    ----------
    file:
        Video file (MP4, MOV, AVI, MKV).
    detect:
        Comma-separated list of additional objects to detect.
    use_gemini_tagger:
        Use Gemini 2.5 Flash for tagging.
    every_n_frames:
        Extract one keyframe every N video frames.
    max_frames:
        Maximum keyframes to process.
    """
    import tempfile

    # Save uploaded video to temp file
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    try:

        detect_objects = None
        if detect:
            detect_objects = [p.strip() for p in detect.split(",") if p.strip()]

        scene_id, scene_graph = analyze_video_full(
            source=tmp_path,
            detect=detect_objects,
            use_gemini_tagger=use_gemini_tagger,
            use_sam3=use_sam3,
            every_n_frames=every_n_frames,
            max_frames=max_frames,
        )
    except Exception as exc:
        logger.exception("Video analysis failed.")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return AnalyzeResponse(
        scene_id=scene_id,
        scene_graph=scene_graph,
        intrinsics_source=scene_graph.calibration.intrinsics_source,
    )


@app.post("/snap", response_model=AnalyzeResponse)
async def snap(file: UploadFile = File(...)):  # noqa: B008
    """Upload an image and get a 3D scene graph. No extra parameters needed.

    Designed for iPhone photos — EXIF metadata is auto-extracted for accurate
    FOV. Just upload the file and get results.
    """
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    try:
        scene_id, scene_graph = analyze_image_full(image)
    except Exception as exc:
        logger.exception("Analysis failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(
        scene_id=scene_id,
        scene_graph=scene_graph,
        intrinsics_source=scene_graph.calibration.intrinsics_source,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a spatial reasoning question about a previously analyzed scene.

    Parameters
    ----------
    request:
        JSON body with ``scene_id`` and ``question``.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Resolve scene_id: use provided or fall back to most recent
    scene_id = request.scene_id
    if scene_id is None:
        if not _scene_cache:
            raise HTTPException(
                status_code=404,
                detail="No scenes available. Upload an image first via /analyze.",
            )
        # Use the most recently cached scene
        scene_id = list(_scene_cache.keys())[-1]

    if scene_id not in _scene_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found. Available: {list(_scene_cache.keys())}",
        )

    _, reasoner, _ = _scene_cache[scene_id]

    try:
        answer = reasoner.ask(request.question)
    except Exception as exc:
        logger.exception("Reasoning failed for scene %s.", scene_id)
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {exc}") from exc

    return AskResponse(answer=answer, scene_id=scene_id)


def _get_debug(scene_id: str) -> DebugArtifacts:
    """Look up debug artifacts for a scene, raising 404 if not found."""
    if scene_id not in _scene_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found.",
        )
    _, _, debug = _scene_cache[scene_id]
    return debug


def _require_jpg(data: bytes | None, scene_id: str, name: str) -> Response:
    """Return a JPEG Response or raise 404 if the artifact is unavailable."""
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' not available for scene '{scene_id}'.",
        )
    return Response(content=data, media_type="image/jpeg")


@app.get("/scene/{scene_id}/detections")
async def scene_detections(scene_id: str):
    """Detection boxes drawn on the image (JPEG)."""
    debug = _get_debug(scene_id)
    return _require_jpg(debug.detections_jpg, scene_id, "detections")


@app.get("/scene/{scene_id}/masks")
async def scene_masks(scene_id: str):
    """Segmentation mask overlay (JPEG)."""
    debug = _get_debug(scene_id)
    return _require_jpg(debug.masks_jpg, scene_id, "masks")


@app.get("/scene/{scene_id}/depth")
async def scene_depth(scene_id: str):
    """Depth heatmap with meter labels (JPEG)."""
    debug = _get_debug(scene_id)
    return _require_jpg(debug.depth_jpg, scene_id, "depth")


@app.get("/scene/{scene_id}/annotated")
async def scene_annotated(scene_id: str):
    """Full annotation: boxes, masks, 3D dimensions, distances (JPEG)."""
    debug = _get_debug(scene_id)
    return _require_jpg(debug.annotated_jpg, scene_id, "annotated")


@app.get("/scene/{scene_id}/pointcloud")
async def scene_pointcloud(scene_id: str):
    """Interactive 3D point cloud (HTML — open in browser)."""
    debug = _get_debug(scene_id)
    if debug.pointcloud_html is None:
        raise HTTPException(
            status_code=404,
            detail=f"'pointcloud' not available for scene '{scene_id}'.",
        )
    return HTMLResponse(content=debug.pointcloud_html)


@app.get("/scene/{scene_id}/floorplan")
async def scene_floorplan(scene_id: str):
    """Top-down 2D floorplan (HTML — open in browser). Multi-view only."""
    debug = _get_debug(scene_id)
    if debug.floorplan_html is None:
        raise HTTPException(
            status_code=404,
            detail=f"Floorplan not available for scene '{scene_id}' (multi-view only).",
        )
    return HTMLResponse(content=debug.floorplan_html)


@app.get("/scene/{scene_id}/tags")
async def scene_tags(scene_id: str):
    """Raw and filtered tags used for detection (JSON)."""
    debug = _get_debug(scene_id)
    return JSONResponse(content={
        "raw_tags": debug.raw_tags,
        "filtered_tags": debug.filtered_tags,
        "anchors_injected": debug.anchors_injected,
    })


@app.get("/scene/{scene_id}/objects")
async def scene_objects(scene_id: str):
    """Post-NMS detection data: label, confidence, bbox (JSON)."""
    debug = _get_debug(scene_id)
    return JSONResponse(content=debug.post_nms_detections or [])


@app.get("/scene/{scene_id}/graph")
async def scene_graph_json(scene_id: str):
    """Full scene graph as JSON."""
    debug = _get_debug(scene_id)
    if debug.scene_graph_json is None:
        raise HTTPException(
            status_code=404,
            detail=f"'graph' not available for scene '{scene_id}'.",
        )
    return JSONResponse(content=debug.scene_graph_json)


@app.get("/scene/{scene_id}/graph/text")
async def scene_graph_text(scene_id: str):
    """Human-readable scene graph table (plain text)."""
    debug = _get_debug(scene_id)
    if debug.scene_graph_text is None:
        raise HTTPException(
            status_code=404,
            detail=f"'graph/text' not available for scene '{scene_id}'.",
        )
    return PlainTextResponse(content=debug.scene_graph_text)


@app.get("/scene/{scene_id}/masks/count")
async def masks_count(scene_id: str):
    """Return number of per-frame mask images available."""
    debug = _get_debug(scene_id)
    count = len(debug.frame_masks_jpg) if debug.frame_masks_jpg else 0
    return {"count": count}


@app.get("/scene/{scene_id}/masks/{frame_idx}")
async def mask_frame(scene_id: str, frame_idx: int):
    """Return per-frame mask overlay image."""
    debug = _get_debug(scene_id)
    if not debug.frame_masks_jpg or frame_idx >= len(debug.frame_masks_jpg):
        raise HTTPException(status_code=404, detail=f"Frame {frame_idx} not available.")
    return Response(content=debug.frame_masks_jpg[frame_idx], media_type="image/jpeg")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "scenes_cached": len(_scene_cache)}


@app.get("/scenes")
async def list_scenes():
    """List cached scene IDs with basic info."""
    scenes = []
    for sid, (sg, _, dbg) in _scene_cache.items():
        scenes.append({
            "id": sid,
            "num_objects": len(sg.objects) if sg else 0,
            "source": dbg.source_name if dbg else None,
        })
    return {"scenes": scenes}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: analyze images or start the API server."""
    parser = argparse.ArgumentParser(
        description="Splat Memory: Single-image spatial reasoning",
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to video file (.mp4/.mov)")
    parser.add_argument("--frames", type=str, help="Path to directory of image frames")
    parser.add_argument(
        "--detect",
        type=str,
        nargs="*",
        help="Additional object categories to detect (merged with auto-discovered tags)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive Q&A loop after analysis",
    )
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for models (default: cuda)",
    )
    parser.add_argument(
        "--no-tagger",
        action="store_true",
        help="Disable RAM++ tagging (use default DINO prompts instead)",
    )
    parser.add_argument(
        "--gemini-tags",
        action="store_true",
        help="Use Gemini 2.5 Flash for tagging instead of RAM++ + Claude filter",
    )
    parser.add_argument(
        "--sam3",
        action="store_true",
        help="Use SAM3 for detection+segmentation instead of Grounding DINO + SAM2",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Override horizontal FOV in degrees (default: auto from EXIF or 70°)",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=30,
        help="Extract one keyframe every N video frames (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Maximum keyframes to process (default: 40)",
    )
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        logger.info("Starting API server on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # --- CLI: video or frames mode ---
    if args.video or args.frames:
        source = args.video or args.frames
        source_path = Path(source)
        if not source_path.exists():
            logger.error("Source not found: %s", source_path)
            sys.exit(1)

        logger.info("Analyzing video/frames: %s", source_path)
        scene_id, scene_graph = analyze_video_full(
            source=str(source_path),
            detect=args.detect,
            use_gemini_tagger=args.gemini_tags,
            use_sam3=args.sam3,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_frames,
        )

        print("\n" + scene_graph.to_prompt_text())
        print(f"\nScene ID: {scene_id}")
        print(f"Objects: {len(scene_graph.objects)}")
        print(f"Relations: {len(scene_graph.relations)}")
        print(f"Scale factor: {scene_graph.calibration.scale_factor:.3f}")
        cal = scene_graph.calibration
        print(f"Frames: {cal.num_frames}")
        print(f"Coordinate frame: {cal.coordinate_frame}")
        if cal.reference_object:
            print(f"Reference object: {cal.reference_object}")
        if cal.scale_warning:
            print(f"WARNING: {cal.scale_warning}")

        # Save debug visualizations to disk
        cached = _scene_cache.get(scene_id)
        debug = cached[2] if cached else None
        if debug:
            out_dir = Path("video_debug_output")
            out_dir.mkdir(exist_ok=True)
            if debug.pointcloud_html:
                (out_dir / "pointcloud.html").write_text(debug.pointcloud_html)
                print(f"Saved: {out_dir / 'pointcloud.html'}")
            if debug.floorplan_html:
                (out_dir / "floorplan.html").write_text(debug.floorplan_html)
                print(f"Saved: {out_dir / 'floorplan.html'}")

    elif args.image:
        # --- CLI: analyze a single image ---
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            sys.exit(1)

        logger.info("Loading image: %s", image_path)
        image = Image.open(image_path)  # Don't convert yet — preserve EXIF

        use_tagger = not args.no_tagger
        scene_id, scene_graph = analyze_image_full(
            image, detect=args.detect, use_tagger=use_tagger, fov_override=args.fov,
        )

        print("\n" + scene_graph.to_prompt_text())
        print(f"\nScene ID: {scene_id}")
        print(f"Objects: {len(scene_graph.objects)}")
        print(f"Relations: {len(scene_graph.relations)}")
        print(f"Scale factor: {scene_graph.calibration.scale_factor:.3f}")
        cal = scene_graph.calibration
        print(f"FOV: {cal.fov_degrees:.1f}° ({cal.intrinsics_source})")
        if cal.reference_object:
            print(f"Reference object: {cal.reference_object}")
    else:
        parser.print_help()
        return

    # --- Interactive Q&A mode ---
    if args.interactive:
        _, reasoner, _ = _scene_cache[scene_id]

        print("\n--- Interactive Q&A Mode ---")
        print("Ask spatial questions about the scene. Type 'quit' or Ctrl+C to exit.\n")

        # Handle SIGINT gracefully
        def _sigint_handler(sig, frame):
            print("\nExiting interactive mode.")
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint_handler)

        while True:
            try:
                question = input("Q: ").strip()
            except EOFError:
                print("\nExiting interactive mode.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Exiting interactive mode.")
                break

            try:
                answer = reasoner.ask(question)
                print(f"\nA: {answer}\n")
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception:
                logger.exception("Reasoning failed.")
                print("Error: could not get answer. See logs for details.\n")


if __name__ == "__main__":
    main()
