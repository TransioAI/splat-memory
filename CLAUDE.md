# Splat Memory — Development Guide

## Project Overview
Single-image spatial reasoning system. Takes one photo, builds a 3D scene graph, and answers spatial questions using Claude.

## Architecture
```
perception/     # Image → detections, masks, depth
  detector.py   # Grounding DINO (fallback: Florence-2)
  segmentor.py  # SAM2 with bbox prompts
  depth.py      # Depth Anything V2 METRIC variant
  pipeline.py   # Orchestrator: detect → segment → depth
fusion/         # 2D+depth → 3D scene
  backproject.py      # Pinhole camera back-projection
  calibration.py      # Intrinsics estimation + auto-scale
  spatial_relations.py # Pairwise spatial relations
scene/          # Data models
  models.py     # Pydantic: SceneObject, SceneRelation, SceneGraph
reasoning/      # LLM spatial Q&A
  llm.py        # Claude-powered spatial reasoning with history
visualization/  # Output rendering
  annotate.py   # OpenCV 2D annotations
  pointcloud.py # Plotly 3D point cloud
main.py         # FastAPI server + CLI entry point
```

## Key Gotchas — READ BEFORE CODING
1. **METRIC depth, not relative**: Use `depth-anything/Depth-Anything-V2-Metric-Indoor-Large`. The relative variant outputs unitless values — useless for measurement.
2. **Resize depth map**: Depth model output size ≠ input image size. Always resize depth to match input image dims before back-projection.
3. **Y-axis is DOWN**: Camera frame convention — Y points downward. "above" = smaller Y, "below" = larger Y.
4. **Median depth, not mean**: Use median for per-object depth to resist outliers from mask edges.
5. **Percentiles for dimensions**: Use 5th/95th percentiles for object width/height/depth, NOT min/max (too noisy).
6. **Auto-calibrate scale**: If a known-size object is detected (door=2.03m, countertop=0.91m, person=1.70m), compute scale_factor = known/estimated and apply globally.
7. **FOV assumption**: Default 70° horizontal FOV (smartphone-like). Estimate intrinsics: fx = width / (2 * tan(fov/2)).

## Models Used
- **Detection**: `IDEA-Research/grounding-dino-base` (primary), Florence-2 (fallback)
- **Segmentation**: `facebook/sam2-hiera-large`
- **Depth**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large`
- **Reasoning**: `claude-sonnet-4-20250514`

## Commands
```bash
# Run API server
python main.py --serve

# Analyze single image
python main.py --image path/to/photo.jpg

# Interactive Q&A
python main.py --image path/to/photo.jpg --interactive
```

## Code Style
- Python 3.10+, type hints everywhere
- Ruff for linting (line-length=100)
- Pydantic v2 for all data models
- No unnecessary abstractions — keep it direct
