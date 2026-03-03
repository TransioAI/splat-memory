# iPhone Camera Guide

Snap a photo with your iPhone, upload it, and get 3D spatial data for every object in the scene.

**Base URL:** `http://<server-ip>:8000`

---

## Quick Start

```bash
# 1. Upload a photo
curl -X POST -F "file=@IMG_1234.heic" http://<server-ip>:8000/snap

# 2. Ask a question (uses the most recent scene automatically)
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How far is the chair from the window?"}'

# 3. Download the annotated image
curl http://<server-ip>:8000/scene/<scene_id>/annotated > annotated.jpg
```

---

## Step 1: Upload

```bash
curl -X POST -F "file=@IMG_1234.heic" http://<server-ip>:8000/snap
```

- Upload the original `.heic` file from your iPhone — do not convert to JPEG
- HEIC preserves EXIF metadata, which gives accurate camera FOV (e.g. 104° for ultrawide)
- JPEG conversions strip EXIF, reducing lateral position accuracy by ~40%

**Response:**

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

Save the `scene_id` — you need it for all follow-up requests.

**Verify EXIF was used:** Check that `intrinsics_source` is `"exif"`. If it says `"default"`, your image had no EXIF data.

---

## Step 2: Ask Questions

```bash
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What objects are in the room?"}'
```

Follow-up questions remember context (up to 50 turns):

```bash
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which one is closest to the camera?"}'
```

To ask about a specific scene:

```bash
curl -X POST http://<server-ip>:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "<scene_id>", "question": "How far is the bed from the wall?"}'
```

---

## Step 3: Get Visualizations

Replace `<scene_id>` with the UUID from step 1.

### Images

```bash
# Full annotation: boxes, masks, 3D dimensions, distances
curl http://<server-ip>:8000/scene/<scene_id>/annotated > annotated.jpg

# Detection boxes only
curl http://<server-ip>:8000/scene/<scene_id>/detections > detections.jpg

# Segmentation masks
curl http://<server-ip>:8000/scene/<scene_id>/masks > masks.jpg

# Depth heatmap with meter labels
curl http://<server-ip>:8000/scene/<scene_id>/depth > depth.jpg
```

### Interactive 3D

Open directly in a browser:

```
http://<server-ip>:8000/scene/<scene_id>/pointcloud
```

### Data

```bash
# Scene graph as JSON
curl http://<server-ip>:8000/scene/<scene_id>/graph

# Scene graph as readable text
curl http://<server-ip>:8000/scene/<scene_id>/graph/text

# Detection data (label, confidence, bbox)
curl http://<server-ip>:8000/scene/<scene_id>/objects

# Tags used for detection
curl http://<server-ip>:8000/scene/<scene_id>/tags
```

---

## Understanding the Response

### Object Fields

| Field | Example | Meaning |
|---|---|---|
| `label` | `"chair"` | What was detected |
| `confidence` | `0.79` | Detection confidence (0 to 1) |
| `position_m` | `[-0.16, 0.34, 5.26]` | 3D position `[x, y, z]` in meters from camera |
| `dimensions_m` | `[0.89, 1.40, 0.91]` | Size `[width, height, depth]` in meters |
| `distance_m` | `5.28` | Distance from camera in meters |

### Spatial Relations

| Predicate | Meaning |
|---|---|
| `left_of` / `right_of` | Horizontal position |
| `above` / `below` | Vertical position |
| `in_front_of` / `behind` | Closer / further from camera |
| `on_top_of` | Above and horizontally close |
| `next_to` | Within 1.5 meters |

### Calibration

| Field | Meaning |
|---|---|
| `fov_degrees` | Camera field of view used |
| `intrinsics_source` | `"exif"` = from photo metadata, `"default"` = 70° fallback |
| `scale_factor` | Scale correction (1.0 = no correction applied) |
| `reference_object` | Object used for scale correction (e.g. `"door"`), or `null` |

---

## iPhone Camera FOV Reference

| Lens | 35mm Equivalent | FOV |
|---|---|---|
| Wide (1x) | 26mm | ~69° |
| Ultrawide (0.5x) | 13mm | ~104° |
| Telephoto (3x) | 77mm | ~26° |

All of these are auto-detected from HEIC EXIF data.

---

## Tips

- **Use HEIC, not JPEG.** AirDrop or iCloud the original file. Screenshots and shared photos are often re-encoded as JPEG without EXIF.
- **1x lens works best** for typical room photos. Ultrawide captures more but objects at edges may have distortion.
- **Better lighting = better detection.** Avoid very dark scenes.
- **Include a door in the frame** if you want accurate absolute measurements. The system uses doors (2.03m standard height) as scale references.
