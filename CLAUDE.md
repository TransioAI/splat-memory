# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Single-image spatial reasoning system. Takes one photo, builds a 3D scene graph, and answers spatial questions using Claude.

## Commands
```bash
# Install (SAM2 requires separate git install)
pip install -e ".[dev]"
pip install git+https://github.com/facebookresearch/sam2.git

# Lint
ruff check .
ruff check --fix .

# Test
pytest
pytest tests/test_foo.py -k "test_name"   # single test

# Run API server (default 0.0.0.0:8000)
python main.py --serve

# Analyze single image
python main.py --image path/to/photo.jpg

# Interactive Q&A after analysis
python main.py --image path/to/photo.jpg --interactive

# Custom detection prompts
python main.py --image photo.jpg --prompts "chair . table . lamp"
```

## Environment Variables
- `ANTHROPIC_API_KEY` — required for reasoning/llm.py (Claude Sonnet 4)
- `DEVICE` — optional, defaults to auto-detect (cuda/cpu)

## Architecture
```
perception/          # Image → detections, masks, depth
  detector.py        # Grounding DINO (fallback: Florence-2)
  segmentor.py       # SAM2 with bbox prompts (fallback: SAM)
  depth.py           # Depth Anything V2 METRIC variant
  pipeline.py        # Orchestrator: detect → segment → depth
fusion/              # 2D+depth → 3D scene
  backproject.py     # Pinhole camera back-projection
  calibration.py     # Intrinsics estimation + auto-scale
  spatial_relations.py # Pairwise spatial relations
scene/
  models.py          # Pydantic v2: SceneObject, SceneRelation, SceneGraph
reasoning/
  llm.py             # Claude-powered spatial Q&A with conversation history
visualization/
  annotate.py        # OpenCV 2D annotations with 3D measurements
  pointcloud.py      # Plotly interactive 3D point cloud
main.py              # FastAPI server + CLI entry point
```

## Data Flow (main.py → analyze_image_full)
1. **Perception** (`pipeline.py`): image → Grounding DINO detections → SAM2 masks → Depth Anything V2 depth map
2. **Calibration** (`calibration.py`): estimate camera intrinsics from 70° FOV assumption
3. **Back-projection** (`backproject.py`): per-object mask + depth → 3D point cloud → median centroid + percentile dimensions
4. **Auto-scale** (`calibration.py`): if a known-size object detected (door=2.03m, person=1.70m, etc.), compute global scale factor and apply to all 3D measurements
5. **Spatial relations** (`spatial_relations.py`): pairwise 3D comparisons → relation predicates (left_of, above, on_top_of, etc.)
6. **Scene graph** (`models.py`): assemble SceneGraph → `to_prompt_text()` for LLM context
7. **Reasoning** (`llm.py`): scene text + user question → Claude Sonnet 4 answer, with conversation history (up to 50 turns)

## Coordinate Frame Convention (cross-module)
- **X** = right, **Y** = down, **Z** = depth (away from camera)
- "above" = smaller Y, "below" = larger Y (Y-axis is DOWN)
- Spatial relation thresholds: DIRECTIONAL=0.2m, ON_TOP_HORIZONTAL=0.5m, NEXT_TO=1.5m

## Key Gotchas — READ BEFORE CODING
1. **METRIC depth, not relative**: Use `depth-anything/Depth-Anything-V2-Metric-Indoor-Large`. The relative variant outputs unitless values — useless for measurement.
2. **Resize depth map**: Depth model output size ≠ input image size. Always resize depth to match input image dims before back-projection.
3. **Median depth, not mean**: Use median for per-object depth to resist outliers from mask edges.
4. **Percentiles for dimensions**: Use 5th/95th percentiles for object width/height/depth, NOT min/max (too noisy).
5. **Auto-calibrate scale**: If a known-size object is detected, compute scale_factor = known/estimated and apply globally.
6. **FOV assumption**: Default 70° horizontal FOV (smartphone-like). Estimate intrinsics: fx = width / (2 * tan(fov/2)).
7. **All ML models lazy-load**: Detector, Segmentor, and DepthEstimator all load weights on first call, not at init.

## Models Used
- **Detection**: `IDEA-Research/grounding-dino-base` (primary), `microsoft/Florence-2-large` (fallback)
- **Segmentation**: `facebook/sam2-hiera-large` (primary), `facebook/sam-vit-large` (fallback)
- **Depth**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`
- **Reasoning**: `claude-sonnet-4-20250514`

## Code Style
- Python 3.10+, type hints everywhere
- Ruff for linting (line-length=100, rules: E, F, I, N, W, UP, B, SIM, RUF)
- Pydantic v2 for all data models
- No unnecessary abstractions — keep it direct
