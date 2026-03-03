# Python SDK Guide

Zero-dependency Python client for the Splat Memory API. No pip install needed — just copy `sdk/` into your project.

---

## Setup

```python
from sdk import SplatMemory

client = SplatMemory("192.168.1.50")        # default port 8000
client = SplatMemory("192.168.1.50", 9000)  # custom port
```

---

## Analyze an Image

### Simple (iPhone / any image)

```python
scene = client.snap("IMG_1234.heic")

print(scene)
# Scene(id='4e5b7228...', objects=15, relations=591, fov=104.2° [exif])

print(scene.scene_id)
# '4e5b7228-b077-4afa-9208-e9c0927b4366'

print(scene.intrinsics_source)
# 'exif'
```

### With Options

```python
scene = client.analyze(
    "photo.jpg",
    detect=["chair", "table", "lamp"],   # extra objects to detect
    fov_degrees=77,                       # override FOV
)

# Or with focal length
scene = client.analyze("photo.jpg", focal_length_35mm=26)
```

---

## Access Detected Objects

```python
for obj in scene.objects:
    print(f"{obj.label}: {obj.distance_m:.1f}m away, conf={obj.confidence:.0%}")
# chair: 5.3m away, conf=79%
# table: 2.6m away, conf=64%
# bed: 2.0m away, conf=64%

# 3D position (meters from camera)
obj = scene.objects[0]
print(f"x={obj.x:.2f}, y={obj.y:.2f}, z={obj.z:.2f}")

# Object dimensions
print(f"width={obj.width:.2f}m, height={obj.height:.2f}m, depth={obj.depth:.2f}m")
```

### Find Objects by Label

```python
chairs = scene.find("chair")
tables = scene.find("table")

# Case-insensitive partial match
computers = scene.find("computer")
# Matches both "computer" and "computer desk"
```

### Nearest / Farthest

```python
closest = scene.nearest()
print(f"{closest.label} at {closest.distance_m:.1f}m")

farthest = scene.farthest()
print(f"{farthest.label} at {farthest.distance_m:.1f}m")
```

---

## Spatial Relations

```python
for rel in scene.relations[:5]:
    print(rel)
# chair left_of table (3.17m)
# chair behind table (3.17m)
# chair left_of bed (3.58m)
# chair above bed (3.58m)
# chair behind bed (3.58m)

# Access fields
rel = scene.relations[0]
print(rel.subject_label, rel.predicate, rel.object_label, rel.distance_m)
```

---

## Ask Questions

```python
answer = client.ask("How far is the chair from the window?")
print(answer)

# Follow-up questions remember context
answer = client.ask("Which object is closest to it?")
print(answer)

# Ask about a specific scene
answer = client.ask("What objects are in the room?", scene_id="4e5b7228-...")
```

---

## Save Visualizations

All save methods use the last analyzed scene by default. Pass `scene_id` to target a specific scene.

```python
# Full annotation: boxes, masks, 3D dimensions, distances
client.save_annotated("annotated.jpg")

# Detection boxes only
client.save_detections("detections.jpg")

# Segmentation masks
client.save_masks("masks.jpg")

# Depth heatmap
client.save_depth("depth.jpg")

# Interactive 3D point cloud (open in browser)
client.save_pointcloud("pointcloud.html")
```

---

## Get Data

```python
# Tags used for detection
tags = client.get_tags()
print(tags["raw_tags"])        # from image tagger
print(tags["filtered_tags"])   # after filtering
print(tags["anchors_injected"])  # spatial anchors added

# Detection data
objects = client.get_objects()
for obj in objects:
    print(f"{obj['label']}: {obj['confidence']:.2f}")

# Full scene graph as dict
graph = client.get_graph()

# Scene graph as readable text
text = client.get_graph_text()
print(text)
```

---

## Calibration Info

```python
cal = scene.calibration

print(f"FOV: {cal.fov_degrees}°")
print(f"Source: {cal.intrinsics_source}")    # exif, user_provided, or default
print(f"Scale: {cal.scale_factor}")
print(f"Reference: {cal.reference_object}")  # e.g. "door" or None
print(f"Image: {cal.image_width}x{cal.image_height}")
```

---

## Health Check

```python
status = client.health()
print(status)
# {'status': 'ok', 'scenes_cached': 3}
```

---

## Error Handling

```python
from sdk import SplatMemory, SplatMemoryError

client = SplatMemory("192.168.1.50")

try:
    scene = client.snap("nonexistent.jpg")
except FileNotFoundError:
    print("Image file not found")

try:
    answer = client.ask("question", scene_id="invalid-id")
except SplatMemoryError as e:
    print(f"API error {e.status}: {e.detail}")
```

---

## Full Example

```python
from sdk import SplatMemory

# Connect
client = SplatMemory("192.168.1.50")

# Analyze
scene = client.snap("IMG_1234.heic")
print(f"Found {len(scene.objects)} objects, FOV={scene.calibration.fov_degrees}°")

# Explore objects
for obj in scene.objects:
    print(f"  {obj.label}: {obj.distance_m:.1f}m away")

# Find specific objects
beds = scene.find("bed")
if beds:
    print(f"\nBed is {beds[0].distance_m:.1f}m from camera")

# Ask questions
answer = client.ask("What is the closest object to the bed?")
print(f"\n{answer}")

# Save outputs
client.save_annotated("annotated.jpg")
client.save_depth("depth.jpg")
client.save_pointcloud("scene.html")

print("\nDone!")
```
