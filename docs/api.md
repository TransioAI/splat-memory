# Splat Memory API Reference

Upload a photo, get back a 3D scene graph with metric positions and dimensions for every detected object, then ask spatial questions in natural language.

**Base URL:** `http://<server-ip>:8000`

---

## POST /analyze

Upload an image and receive a 3D scene graph.

**Content-Type:** `multipart/form-data`

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Image file (JPEG, PNG, HEIC/HEIF). iPhone HEIC images are supported natively — EXIF metadata is auto-extracted for accurate FOV. |
| `detect` | string | No | Comma-separated additional object categories to detect (e.g. `"chair,table,lamp"`). These are merged with auto-discovered tags from the image, so both automatic and user-specified objects are detected. |
| `fov_degrees` | float | No | Horizontal field-of-view override in degrees. Takes priority over EXIF metadata. |
| `focal_length_35mm` | float | No | 35mm-equivalent focal length in mm. Converted to FOV if `fov_degrees` is not provided. |

### FOV Resolution

The system resolves camera FOV in this order:
1. `fov_degrees` parameter → `intrinsics_source: "user_provided"`
2. `focal_length_35mm` parameter (converted to FOV) → `intrinsics_source: "user_provided"`
3. EXIF metadata from image (`FocalLengthIn35mmFilm`) → `intrinsics_source: "exif"`
4. Default: 70 degrees → `intrinsics_source: "default"`

> **Tip:** Upload original HEIC files from iPhone for best accuracy. JPEG conversions often strip EXIF metadata, causing the system to fall back to 70° default which can introduce ~40% error in lateral positions for ultrawide lenses.

### Examples

```bash
# Basic usage
curl -X POST -F "file=@photo.jpg" http://<server-ip>:8000/analyze

# With additional objects to detect (merged with auto-discovered tags)
curl -X POST -F "file=@photo.jpg" -F "detect=chair,table,lamp" http://<server-ip>:8000/analyze

# With FOV override
curl -X POST -F "file=@photo.jpg" -F "fov_degrees=77" http://<server-ip>:8000/analyze

# With 35mm focal length (converted to FOV automatically)
curl -X POST -F "file=@photo.jpg" -F "focal_length_35mm=26" http://<server-ip>:8000/analyze

# iPhone HEIC image (EXIF auto-extracted)
curl -X POST -F "file=@IMG_1234.heic" http://<server-ip>:8000/analyze
```

### Response

```json
{
  "scene_id": "4e5b7228-b077-4afa-9208-e9c0927b4366",
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

### Response Fields

| Field | Description |
|---|---|
| `scene_id` | UUID to reference this scene in all other endpoints. |
| `intrinsics_source` | How FOV was determined: `"user_provided"`, `"exif"`, or `"default"`. |
| `scene_graph.objects[].id` | Unique object ID within this scene. |
| `scene_graph.objects[].label` | Detected object category (e.g. `"chair"`, `"door"`, `"wall"`). |
| `scene_graph.objects[].confidence` | Detection confidence score (0.0 to 1.0). |
| `scene_graph.objects[].bbox` | 2D bounding box in pixels: `[x_min, y_min, x_max, y_max]`. |
| `scene_graph.objects[].position_m` | 3D centroid `[x, y, z]` in meters from camera. |
| `scene_graph.objects[].dimensions_m` | Object size `[width, height, depth]` in meters. |
| `scene_graph.objects[].distance_m` | Euclidean distance from camera in meters. |
| `scene_graph.relations[].subject_id` | ID of the subject object. |
| `scene_graph.relations[].predicate` | Spatial relation (see Spatial Predicates below). |
| `scene_graph.relations[].object_id` | ID of the object being related to. |
| `scene_graph.relations[].distance_m` | Distance between the two objects in meters. |
| `scene_graph.calibration.fov_degrees` | Horizontal FOV used for 3D reconstruction. |
| `scene_graph.calibration.scale_factor` | Scale correction applied (1.0 = no reference object found). |
| `scene_graph.calibration.reference_object` | Object used for scale calibration (e.g. `"door"`), or `null`. |

### Spatial Predicates

| Predicate | Meaning |
|---|---|
| `left_of` | Subject is to the left of object (X-axis) |
| `right_of` | Subject is to the right of object |
| `above` | Subject is higher than object (smaller Y) |
| `below` | Subject is lower than object |
| `in_front_of` | Subject is closer to camera (smaller Z) |
| `behind` | Subject is further from camera |
| `on_top_of` | Subject is above and horizontally close |
| `next_to` | Objects are within 1.5 meters of each other |

---

## POST /ask

Ask a natural-language spatial question about a previously analyzed scene.

**Content-Type:** `application/json`

### Parameters

| Field | Type | Required | Description |
|---|---|---|---|
| `scene_id` | string | No | Scene UUID from `/analyze`. If omitted, uses the most recently analyzed scene. |
| `question` | string | Yes | Natural-language spatial question. |

### Examples

```bash
# Ask about the most recent scene
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How far is the chair from the window?"}'

# Ask about a specific scene
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "4e5b7228-...", "question": "What is to the left of the bed?"}'
```

### Response

```json
{
  "answer": "The chair is approximately 2.95 meters from the window...",
  "scene_id": "4e5b7228-b077-4afa-9208-e9c0927b4366"
}
```

### Conversation History

The system maintains conversation history per scene (up to 50 turns), so follow-up questions work:

```bash
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What objects are in the room?"}'

curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which one is closest to the camera?"}'
```

---

## Scene Output Endpoints

After analyzing an image, use the `scene_id` from the response to retrieve visualizations and data.

### GET /scene/{scene_id}/detections

Detection boxes drawn on the image.

**Content-Type:** `image/jpeg`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/detections > detections.jpg
```

---

### GET /scene/{scene_id}/masks

Segmentation mask overlay.

**Content-Type:** `image/jpeg`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/masks > masks.jpg
```

---

### GET /scene/{scene_id}/depth

Depth heatmap with meter labels (Inferno colormap).

**Content-Type:** `image/jpeg`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/depth > depth.jpg
```

---

### GET /scene/{scene_id}/annotated

Full annotation: bounding boxes, segmentation masks, 3D dimensions, and distances.

**Content-Type:** `image/jpeg`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/annotated > annotated.jpg
```

---

### GET /scene/{scene_id}/pointcloud

Interactive 3D point cloud visualization. Open in a browser.

**Content-Type:** `text/html`

```bash
# Open directly in browser
open "http://<server-ip>:8000/scene/{scene_id}/pointcloud"

# Or save to file
curl http://<server-ip>:8000/scene/{scene_id}/pointcloud > pointcloud.html
```

---

### GET /scene/{scene_id}/tags

Raw and filtered tags used for object detection.

**Content-Type:** `application/json`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/tags
```

**Response:**
```json
{
  "raw_tags": ["bed", "chair", "computer", "lamp", "table", "window"],
  "filtered_tags": ["bed", "chair", "computer", "lamp", "table"],
  "anchors_injected": ["ceiling", "countertop", "door", "floor", "shelf", "wall", "window"]
}
```

---

### GET /scene/{scene_id}/objects

Post-NMS detection data for all detected objects.

**Content-Type:** `application/json`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/objects
```

**Response:**
```json
[
  {
    "label": "chair",
    "confidence": 0.79,
    "bbox": [1502.3, 604.8, 2080.1, 1578.2]
  }
]
```

---

### GET /scene/{scene_id}/graph

Full scene graph as JSON (same as `scene_graph` in `/analyze` response).

**Content-Type:** `application/json`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/graph
```

---

### GET /scene/{scene_id}/graph/text

Human-readable scene graph table.

**Content-Type:** `text/plain`

```bash
curl http://<server-ip>:8000/scene/{scene_id}/graph/text
```

**Response:**
```
=== SCENE GRAPH ===

Image: 4032x2268 px
FOV: 104.25 degrees (source: exif)
Scale factor: 1.000

--- OBJECTS ---
ID   Label       Conf   Position (x,y,z) m           Dimensions (w,h,d) m    Dist m
0    chair       0.79   ( -0.16,  +0.34,  +5.26)     ( 0.89, 1.40, 0.91)     5.28
...
```

---

## GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "scenes_cached": 3
}
```

| Field | Description |
|---|---|
| `status` | `"ok"` if server is running. |
| `scenes_cached` | Number of analyzed scenes currently held in memory. |

---

## Coordinate System

All 3D measurements use a camera-centered coordinate frame. All units are in **meters**.

```
        +Y (down)
        |
        |
        +-------> +X (right)
       /
      /
    +Z (depth / away from camera)
```

| Direction | Axis | Value |
|---|---|---|
| Right | X | positive |
| Left | X | negative |
| Down | Y | positive |
| Up | Y | negative |
| Further from camera | Z | positive |
| Closer to camera | Z | negative |

---

## Scale Calibration

If the system detects a known-size reference object, it computes a global scale correction factor applied to all 3D measurements.

| Reference Object | Known Height | Min Detection Confidence |
|---|---|---|
| `door` | 2.03m | 50% |
| `doorway` | 2.03m | 50% |
| `countertop` | 0.91m | 50% |
| `counter` | 0.91m | 50% |

When no reference object is detected, `scale_factor` is `1.0` and `reference_object` is `null`.

---

## Error Responses

All errors return JSON with a `detail` field:

```json
{"detail": "Scene 'abc-123' not found."}
```

| Status | Cause |
|---|---|
| 400 | Invalid file type (not an image) |
| 404 | Scene ID not found |
| 422 | Missing required parameter |
| 500 | Internal server error |
