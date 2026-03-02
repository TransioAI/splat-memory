"""FastAPI server and CLI entry point for splat-memory."""

from __future__ import annotations

import argparse
import io
import logging
import signal
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from reasoning.llm import SpatialReasoner
from scene.models import CalibrationInfo, SceneGraph, SceneObject, SceneRelation

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

# In-memory scene cache: scene_id -> (SceneGraph, SpatialReasoner)
_scene_cache: dict[str, tuple[SceneGraph, SpatialReasoner]] = {}

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    scene_id: str
    scene_graph: SceneGraph


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


def get_pipeline():
    """Lazily initialise the perception pipeline (heavy model loading)."""
    global _pipeline
    if _pipeline is None:
        from perception.pipeline import PerceptionPipeline

        _pipeline = PerceptionPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def analyze_image_full(
    image: Image.Image,
    text_prompts: list[str] | None = None,
) -> tuple[str, SceneGraph]:
    """Run the full perception + fusion pipeline and build a SceneGraph.

    Steps:
        1. Run perception pipeline (detect, segment, depth).
        2. Estimate camera intrinsics from image dimensions.
        3. Back-project each detection + mask to 3D.
        4. Auto-calibrate scale using known-size reference objects.
        5. Apply scale correction.
        6. Compute pairwise spatial relations.
        7. Assemble SceneGraph.
        8. Cache with a UUID scene_id.

    Parameters
    ----------
    image:
        An RGB PIL image.
    text_prompts:
        Optional list of object categories to detect beyond the defaults.

    Returns
    -------
    tuple[str, SceneGraph]
        ``(scene_id, scene_graph)``
    """
    from fusion.backproject import backproject_to_3d
    from fusion.calibration import apply_scale, auto_calibrate_scale, estimate_intrinsics
    from fusion.spatial_relations import compute_spatial_relations

    pipeline = get_pipeline()
    width, height = image.size

    # 1. Perception: detect -> segment -> depth
    logger.info("Running perception pipeline on %dx%d image...", width, height)
    result = pipeline.run(image, text_prompts=text_prompts)
    detections = result.detections
    masks = result.masks
    depth_map = result.depth_map

    logger.info("Perception complete: %d detections.", len(detections))

    # 2. Estimate camera intrinsics
    intrinsics = estimate_intrinsics(width, height)

    # 3. Back-project each detection to 3D
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

    # 4. Auto-calibrate scale
    all_obj3d = [obj for _, obj in objects_3d]
    scale_factor = auto_calibrate_scale(all_obj3d)

    # 5. Apply scale correction
    if scale_factor != 1.0:
        apply_scale(all_obj3d, scale_factor)

    # Find reference object label (first match from known sizes)
    from fusion.calibration import KNOWN_SIZES

    reference_object: str | None = None
    for obj in all_obj3d:
        label_lower = obj.label.lower()
        if any(k in label_lower or label_lower in k for k in KNOWN_SIZES):
            reference_object = obj.label
            break

    # 6. Compute spatial relations
    spatial_rels = compute_spatial_relations(all_obj3d)

    # 7. Assemble SceneGraph
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
        fov_degrees=70.0,
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

    # 8. Cache
    scene_id = str(uuid.uuid4())
    reasoner = SpatialReasoner()
    reasoner.set_scene(scene_graph)
    _scene_cache[scene_id] = (scene_graph, reasoner)

    logger.info(
        "Scene %s built: %d objects, %d relations, scale=%.3f",
        scene_id,
        len(scene_objects),
        len(scene_relations),
        scale_factor,
    )

    return scene_id, scene_graph


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),  # noqa: B008
    prompts: str | None = Form(None),
):
    """Upload an image and get back a scene graph with 3D spatial info.

    Parameters
    ----------
    file:
        Image file (JPEG, PNG, etc.).
    prompts:
        Optional comma-separated list of additional object categories to detect.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    text_prompts: list[str] | None = None
    if prompts:
        text_prompts = [p.strip() for p in prompts.split(",") if p.strip()]

    try:
        scene_id, scene_graph = analyze_image_full(image, text_prompts=text_prompts)
    except Exception as exc:
        logger.exception("Analysis failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(scene_id=scene_id, scene_graph=scene_graph)


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

    _, reasoner = _scene_cache[scene_id]

    try:
        answer = reasoner.ask(request.question)
    except Exception as exc:
        logger.exception("Reasoning failed for scene %s.", scene_id)
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {exc}") from exc

    return AskResponse(answer=answer, scene_id=scene_id)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "scenes_cached": len(_scene_cache)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: analyze images or start the API server."""
    parser = argparse.ArgumentParser(
        description="Splat Memory: Single-image spatial reasoning",
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        help="Additional object categories to detect",
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
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        logger.info("Starting API server on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.image:
        parser.print_help()
        return

    # --- CLI: analyze a single image ---
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    logger.info("Loading image: %s", image_path)
    image = Image.open(image_path).convert("RGB")

    scene_id, scene_graph = analyze_image_full(image, text_prompts=args.prompts)

    # Print results
    print("\n" + scene_graph.to_prompt_text())
    print(f"\nScene ID: {scene_id}")
    print(f"Objects: {len(scene_graph.objects)}")
    print(f"Relations: {len(scene_graph.relations)}")
    print(f"Scale factor: {scene_graph.calibration.scale_factor:.3f}")
    if scene_graph.calibration.reference_object:
        print(f"Reference object: {scene_graph.calibration.reference_object}")

    # --- Interactive Q&A mode ---
    if args.interactive:
        _, reasoner = _scene_cache[scene_id]

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
