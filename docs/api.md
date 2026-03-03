# Splat Memory API Reference

## Overview

Splat Memory provides a REST API for single-image 3D spatial reasoning. Upload a photo, get back a scene graph with metric 3D positions, then ask spatial questions about the scene.

**Base URL:** `http://localhost:8000` (default)

Start the server:
```bash
python main.py --serve --port 8000
```

---

## Endpoints

### POST /analyze

Upload an image and receive a 3D scene graph.

**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Image file (JPEG, PNG, HEIC/HEIF). HEIC files from iPhone are supported natively. |
| `prompts` | string | No | Comma-separated object categories to detect (e.g. `"chair,table,lamp"`). When provided, RAM++ tagging is skipped. |
| `fov_degrees` | float | No | Horizontal FOV override in degrees. Takes priority over EXIF metadata. |
| `focal_length_35mm` | float | No | 35mm-equivalent focal length in mm. Converted to FOV if `fov_degrees` is not provided. |

**FOV Resolution Priority:**
1. `fov_degrees` parameter (if provided)
2. `focal_length_35mm` parameter (converted to FOV)
3. EXIF metadata from image (`FocalLengthIn35mmFilm`)
4. Default: 70 degrees (with warning logged)

**Example:**
```bash
# Basic usage
curl -X POST -F "file=@photo.jpg" http://localhost:8000/analyze

# With custom prompts (skips RAM++ tagger)
curl -X POST -F "file=@photo.jpg" -F "prompts=chair,table,lamp" http://localhost:8000/analyze

# With FOV override
curl -X POST -F "file=@photo.jpg" -F "fov_degrees=77" http://localhost:8000/analyze

# iPhone HEIC image (EXIF auto-extracted)
curl -X POST -F "file=@IMG_1234.heic" http://localhost:8000/analyze
```

**Response:**
```json
{
  "scene_id": "4e5b7228-b077-4afa-9208-e9c0927b4366",
  "debug_url": "/debug/4e5b7228-b077-4afa-9208-e9c0927b4366",
  "intrinsics_source": "exif",
  "scene_graph": {
    "objects": [
      {
        "id": 0,
        "label": "chair",
        "confidence": 0.79,
        "bbox": [1502.3, 604.8, 2080.1, 1578.2],
        "position_m": [-0.16, 0.34, 5.26],
        "dimensions_m": [0.89, 1.40, 0.91],
        "distance_m": 5.28
      }
    ],
    "relations": [
      {
        "subject_id": 0,
        "subject_label": "chair",
        "predicate": "left_of",
        "object_id": 1,
        "object_label": "table",
        "distance_m": 3.45
      }
    ],
    "calibration": {
      "fov_degrees": 104.25,
      "intrinsics_source": "exif",
      "scale_factor": 1.0,
      "reference_object": null,
      "image_width": 4032,
      "image_height": 2268
    }
  }
}
```

**Response Fields:**
- `scene_id` — UUID to reference this scene in `/ask` and `/debug` endpoints.
- `debug_url` — Path to debug visualization artifacts for this scene.
- `intrinsics_source` — How FOV was determined: `"user_provided"`, `"exif"`, or `"default"`.
- `scene_graph.objects[].position_m` — 3D centroid `(x, y, z)` in meters. X=right, Y=down, Z=depth.
- `scene_graph.objects[].dimensions_m` — Object size `(width, height, depth)` in meters.
- `scene_graph.objects[].distance_m` — Distance from camera in meters.
- `scene_graph.calibration.scale_factor` — Applied scale correction (1.0 = no reference object found).

---

### POST /ask

Ask a spatial reasoning question about a previously analyzed scene.

**Content-Type:** `application/json`

| Field | Type | Required | Description |
|---|---|---|---|
| `scene_id` | string | No | Scene UUID from `/analyze`. If omitted, uses the most recently analyzed scene. |
| `question` | string | Yes | Natural-language spatial question. |

**Example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How far is the chair from the window?"}'

# With explicit scene_id
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "4e5b7228-...", "question": "What is to the left of the bed?"}'
```

**Response:**
```json
{
  "answer": "The chair is approximately 2.95 meters from the window...",
  "scene_id": "4e5b7228-b077-4afa-9208-e9c0927b4366"
}
```

Conversation history is maintained per scene (up to 50 turns), so follow-up questions work:
```bash
curl -X POST http://localhost:8000/ask \
  -d '{"question": "What objects are in the room?"}'

curl -X POST http://localhost:8000/ask \
  -d '{"question": "Which one is closest to the camera?"}'
```

---

### GET /debug/{scene_id}

List available debug artifacts for a scene.

**Example:**
```bash
curl http://localhost:8000/debug/4e5b7228-b077-4afa-9208-e9c0927b4366
```

**Response:**
```json
{
  "scene_id": "4e5b7228-b077-4afa-9208-e9c0927b4366",
  "artifacts": [
    "raw_tags", "filtered_tags", "detections_json",
    "scene_graph", "scene_graph_text",
    "detections", "masks", "depth", "annotated", "pointcloud"
  ]
}
```

---

### GET /debug/{scene_id}/{artifact_name}

Retrieve a specific debug artifact. Returns the correct Content-Type automatically.

| Artifact | Content-Type | Description |
|---|---|---|
| `raw_tags` | application/json | RAM++ raw tags from the image |
| `filtered_tags` | application/json | Claude-filtered tags + injected spatial anchors |
| `detections_json` | application/json | Post-NMS detection data (label, confidence, bbox) |
| `scene_graph` | application/json | Full scene graph as JSON |
| `scene_graph_text` | text/plain | Human-readable scene graph table |
| `detections` | image/jpeg | Post-NMS detection boxes drawn on image |
| `masks` | image/jpeg | SAM2 segmentation mask overlay |
| `depth` | image/jpeg | Depth heatmap (Inferno colormap with meter labels) |
| `annotated` | image/jpeg | Full annotation: boxes, masks, 3D dimensions, distances |
| `pointcloud` | text/html | Interactive 3D Plotly point cloud (open in browser) |

**Examples:**
```bash
# Download annotated image
curl http://localhost:8000/debug/{scene_id}/annotated > annotated.jpg

# View depth heatmap
curl http://localhost:8000/debug/{scene_id}/depth > depth.jpg

# Get filtered tags as JSON
curl http://localhost:8000/debug/{scene_id}/filtered_tags

# Open interactive 3D point cloud in browser
open "http://localhost:8000/debug/{scene_id}/pointcloud"

# Get scene graph as readable text
curl http://localhost:8000/debug/{scene_id}/scene_graph_text
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "scenes_cached": 3
}
```

---

## CLI Usage

```bash
# Analyze a single image
python main.py --image photo.jpg

# With interactive Q&A
python main.py --image photo.jpg --interactive

# Custom detection prompts (skips RAM++ tagger)
python main.py --image photo.jpg --prompts "chair" "table" "lamp"

# Override FOV
python main.py --image photo.jpg --fov 77

# Disable RAM++ tagger (use default DINO prompts)
python main.py --image photo.jpg --no-tagger

# Start API server
python main.py --serve --port 8000 --device cuda
```

---

## Camera Intrinsics

The system needs horizontal FOV to compute camera intrinsics (focal length). Accuracy of lateral (X-axis) positions depends directly on FOV accuracy.

| Source | How | Accuracy |
|---|---|---|
| EXIF metadata | Automatic from `FocalLengthIn35mmFilm` | Best (exact lens data) |
| User-provided | `fov_degrees` or `focal_length_35mm` param | Good (if user knows their camera) |
| Default (70 deg) | Fallback when no metadata available | Approximate (typical smartphone) |

**Common smartphone FOVs:**
| Camera | 35mm Equivalent | FOV |
|---|---|---|
| iPhone 13 Wide (1x) | 26mm | ~69 deg |
| iPhone 13 Ultrawide (0.5x) | 13mm | ~104 deg |
| iPhone 13 Telephoto (3x) | 77mm | ~26 deg |
| Samsung Galaxy S23 Wide | 23mm | ~76 deg |
| Google Pixel 7 Wide | 25mm | ~72 deg |

**Tip:** iPhone HEIC photos contain full EXIF data and FOV is extracted automatically. JPEG conversions may strip EXIF — upload the original file when possible.

---

## Coordinate System

All 3D measurements use a camera-centered coordinate frame:

- **X-axis** = right (positive = right side of image)
- **Y-axis** = down (positive = lower in image)
- **Z-axis** = depth (positive = further from camera)

This means:
- "above" = smaller Y value
- "below" = larger Y value
- "left" = smaller X value
- "right" = larger X value
- "closer" = smaller Z value
- "further" = larger Z value

All units are in **meters**.
