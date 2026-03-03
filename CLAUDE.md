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

# Additional objects to detect (merged with auto-discovered tags)
python main.py --image photo.jpg --detect "chair" "table" "lamp"

# SoM pipeline: SAM2 auto-mask + Gemini labeling (requires GEMINI_API_KEY)
python main.py --image photo.jpg --som
```

## Environment Variables
- `ANTHROPIC_API_KEY` — required for reasoning/llm.py (Claude Sonnet 4)
- `GEMINI_API_KEY` — required for SoM pipeline (Gemini 2.5 Flash labeling)
- `DEVICE` — optional, defaults to auto-detect (cuda/cpu)

## Architecture
```
perception/          # Image → detections, masks, depth
  tagger.py          # RAM++ image tagging (auto-discover objects)
  tag_filter.py      # Claude-based tag filtering (remove non-physical tags)
  detector.py        # Grounding DINO per-tag detection (fallback: Florence-2)
  segmentor.py       # SAM2 with bbox prompts (fallback: SAM)
  auto_mask.py       # SAM2 automatic mask generation (SoM pipeline)
  som_labeler.py     # SoM: numbered markers + Gemini 2.5 Flash labeling
  depth.py           # Depth Anything V2 METRIC variant
  pipeline.py        # Orchestrator: tag → detect → segment → depth (+ SoM mode)
fusion/              # 2D+depth → 3D scene
  backproject.py     # Pinhole camera back-projection
  calibration.py     # EXIF FOV extraction + intrinsics + auto-scale
  spatial_relations.py # Pairwise spatial relations
scene/
  models.py          # Pydantic v2: SceneObject, SceneRelation, SceneGraph
reasoning/
  llm.py             # Claude-powered spatial Q&A with conversation history
visualization/
  annotate.py        # OpenCV 2D annotations with 3D measurements
  pointcloud.py      # Plotly interactive 3D point cloud
sdk/
  client.py          # Zero-dependency Python SDK client
main.py              # FastAPI server + CLI entry point
docs/                # API reference, iPhone guide, SDK guide, pipeline diagram
```

## Data Flow (main.py → analyze_image_full)
1. **EXIF extraction** (`calibration.py`): extract FOV from image EXIF before RGB conversion (priority: user override > EXIF > 70° default)
2. **Tagging** (`tagger.py` → `tag_filter.py`): RAM++ discovers tags → Claude filters non-physical ones → merge user `detect` objects + spatial anchors
3. **Detection** (`detector.py`): per-tag Grounding DINO with confidence overrides → cross-tag NMS
4. **Segmentation** (`segmentor.py`): SAM2 masks from detection bboxes
5. **Depth** (`depth.py`): Depth Anything V2 metric depth map
6. **Back-projection** (`backproject.py`): per-object mask + depth → 3D point cloud → median centroid + P5/P95 dimensions
7. **Auto-scale** (`calibration.py`): if door (2.03m) or countertop (0.91m) detected at >=50% confidence, apply global scale correction
8. **Spatial relations** (`spatial_relations.py`): pairwise 3D centroid comparisons → relation predicates
9. **Scene graph** (`models.py`): assemble SceneGraph → `to_prompt_text()` for LLM context
10. **Debug artifacts**: render annotated image, masks, depth heatmap, point cloud (in-memory)
11. **Reasoning** (`llm.py`): scene text + user question → Claude Sonnet 4 answer (up to 50 turns)

## SoM Pipeline (alternative)
**Set-of-Mark**: segment first, then label — flips the standard detect-then-segment approach.
- **Flow**: SAM2 auto-mask → numbered marker overlay → Gemini 2.5 Flash labels each segment → depth → 3D
- **When to use**: When tag-based detection mislabels objects (e.g. kitchen counter as "desk")
- **CLI**: `python main.py --image photo.jpg --som`
- **API**: `POST /analyze` with `use_som=true`
- **SDK**: `client.analyze("photo.jpg", use_som=True)`
- Requires `GEMINI_API_KEY` env var

## API Endpoints
- `POST /snap` — simple image upload (EXIF auto-extracted)
- `POST /analyze` — image upload with options (detect, fov_degrees, focal_length_35mm, use_som)
- `POST /ask` — spatial Q&A with conversation history
- `GET /scene/{id}/detections|masks|depth|annotated|pointcloud|som` — image outputs
- `GET /scene/{id}/tags|objects|graph|graph/text` — data outputs
- `GET /health` — server health check

## Coordinate Frame Convention (cross-module)
- **X** = right, **Y** = down, **Z** = depth (away from camera)
- "above" = smaller Y, "below" = larger Y (Y-axis is DOWN)
- Spatial relation thresholds: DIRECTIONAL=0.2m, ON_TOP_HORIZONTAL=0.5m, NEXT_TO=1.5m

## Key Gotchas — READ BEFORE CODING
1. **METRIC depth, not relative**: Use `depth-anything/Depth-Anything-V2-Metric-Indoor-Large`. The relative variant outputs unitless values — useless for measurement.
2. **Resize depth map**: Depth model output size ≠ input image size. Always resize depth to match input image dims before back-projection.
3. **Median depth, not mean**: Use median for per-object depth to resist outliers from mask edges.
4. **Percentiles for dimensions**: Use 5th/95th percentiles for object width/height/depth, NOT min/max (too noisy).
5. **Auto-calibrate scale**: Only door/doorway (2.03m) and countertop/counter (0.91m) are used as references — must be detected at >=50% confidence.
6. **EXIF before RGB**: Extract FOV from EXIF metadata BEFORE calling `.convert("RGB")` which strips metadata.
7. **FOV resolution**: User-provided > EXIF > 70° default. Estimate intrinsics: fx = width / (2 * tan(fov/2)).
8. **All ML models lazy-load**: Detector, Segmentor, DepthEstimator, RAM++ tagger all load weights on first call, not at init.
9. **HEIC support**: `pillow-heif` registered at module level. PIL detects format by magic bytes, not file extension.

## Models Used
- **Tagging**: `xinyu1205/recognize-anything-plus-model` (RAM++)
- **Tag Filter**: Claude (via `perception/tag_filter.py`)
- **Detection**: `IDEA-Research/grounding-dino-base` (primary), `microsoft/Florence-2-large` (fallback)
- **Segmentation**: `facebook/sam2-hiera-large` (primary), `facebook/sam-vit-large` (fallback)
- **Auto-mask**: `facebook/sam2-hiera-large` via `pipeline("mask-generation")` (SoM mode)
- **SoM Labeling**: `gemini-2.5-flash` via `google-genai` (SoM mode)
- **Depth**: `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`
- **Reasoning**: `claude-sonnet-4-20250514`

## Code Style
- Python 3.10+, type hints everywhere
- Ruff for linting (line-length=100, rules: E, F, I, N, W, UP, B, SIM, RUF)
- Pydantic v2 for all data models
- No unnecessary abstractions — keep it direct
