"""Run the full pipeline with stage-by-stage output saving for debugging."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline-debug")

OUTPUT_DIR = Path("debug_output")


def save_stage_header(stage_num: int, name: str) -> float:
    """Print a stage header and return the start time."""
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: {name}")
    print(f"{'='*60}")
    return time.time()


def save_stage_footer(start: float) -> None:
    """Print elapsed time for a stage."""
    elapsed = time.time() - start
    print(f"  ⏱  {elapsed:.1f}s")


def draw_detections(
    image: Image.Image,
    detections: list,
    output_path: str,
    title: str = "",
) -> None:
    """Draw bounding boxes and labels on image and save."""
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
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

    if title:
        cv2.putText(canvas, title, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"  Saved: {output_path}")


def save_depth_visualization(depth_map: np.ndarray, output_path: str) -> None:
    """Save depth map as a colorized heatmap."""
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        cv2.imwrite(output_path, np.zeros_like(depth_map, dtype=np.uint8))
        return

    d_min, d_max = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    normalized = np.clip((depth_map - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Add scale bar text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colored, f"{d_min:.1f}m", (10, 30), font, 0.7, (255, 255, 255), 2)
    h = colored.shape[0]
    cv2.putText(colored, f"{d_max:.1f}m", (10, h - 15), font, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, colored)
    print(f"  Saved: {output_path}")


def save_masks_visualization(
    image: Image.Image,
    detections: list,
    masks: list[np.ndarray],
    output_path: str,
) -> None:
    """Save overlay of all masks on the image."""
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

        # Label at mask centroid
        ys, xs = np.where(bool_mask)
        if len(ys) > 0:
            cx, cy = int(np.median(xs)), int(np.median(ys))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                canvas, det.label, (cx, cy),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

    cv2.imwrite(output_path, canvas.astype(np.uint8))
    print(f"  Saved: {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline_debug.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Image: {image_path} ({width}x{height})")

    # =====================================================================
    # STAGE 1: RAM++ Tagging
    # =====================================================================
    t = save_stage_header(1, "RAM++ IMAGE TAGGING")

    from perception.tagger import ImageTagger

    tagger = ImageTagger(device="cuda")
    raw_tags = tagger.tag(image)

    print(f"  Raw tags ({len(raw_tags)}): {raw_tags}")

    with open(OUTPUT_DIR / "01_ram_raw_tags.json", "w") as f:
        json.dump({"raw_tags": raw_tags, "count": len(raw_tags)}, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / '01_ram_raw_tags.json'}")
    save_stage_footer(t)

    # =====================================================================
    # STAGE 2: Claude Tag Filter
    # =====================================================================
    t = save_stage_header(2, "CLAUDE TAG FILTER")

    from perception.tag_filter import TagFilter

    tag_filter = TagFilter()
    filtered_tags = tag_filter.filter_tags(raw_tags)

    print(f"  Filtered tags ({len(filtered_tags)}): {filtered_tags}")
    print(f"  Removed: {set(raw_tags) - set(filtered_tags)}")

    # Inject spatial anchors
    from perception.pipeline import SPATIAL_ANCHORS

    anchors_added = []
    for anchor in sorted(SPATIAL_ANCHORS):
        if anchor not in filtered_tags:
            filtered_tags.append(anchor)
            anchors_added.append(anchor)
    if anchors_added:
        print(f"  Spatial anchors injected: {anchors_added}")
    print(f"  Final tags ({len(filtered_tags)}): {filtered_tags}")

    with open(OUTPUT_DIR / "02_filtered_tags.json", "w") as f:
        json.dump({
            "raw_tags": raw_tags,
            "filtered_tags": filtered_tags,
            "anchors_injected": anchors_added,
            "removed": list(set(raw_tags) - set(filtered_tags)),
        }, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / '02_filtered_tags.json'}")
    save_stage_footer(t)

    # =====================================================================
    # STAGE 3: Per-Tag Grounding DINO Detection
    # =====================================================================
    t = save_stage_header(3, "PER-TAG GROUNDING DINO")

    from perception.detector import ObjectDetector

    detector = ObjectDetector(device="cuda", confidence_threshold=0.3)

    all_per_tag: dict[str, list] = {}
    all_detections = []
    for tag in filtered_tags:
        dets = detector.detect(image, text_prompts=[tag])
        all_per_tag[tag] = [
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in dets
        ]
        all_detections.extend(dets)
        print(f"  Tag '{tag}': {len(dets)} detections")

    with open(OUTPUT_DIR / "03_per_tag_detections.json", "w") as f:
        json.dump(all_per_tag, f, indent=2)

    # Save pre-NMS visualization
    draw_detections(
        image, all_detections,
        str(OUTPUT_DIR / "03_detections_pre_nms.jpg"),
        title=f"Pre-NMS: {len(all_detections)} detections",
    )
    save_stage_footer(t)

    # =====================================================================
    # STAGE 4: Cross-Category NMS
    # =====================================================================
    t = save_stage_header(4, "CROSS-CATEGORY NMS")

    from perception.nms import cross_category_nms

    nms_detections = cross_category_nms(all_detections, iou_threshold=0.5)

    print(f"  Before NMS: {len(all_detections)}")
    print(f"  After NMS:  {len(nms_detections)}")
    for d in nms_detections:
        print(f"    - {d.label} ({d.confidence:.0%})")

    draw_detections(
        image, nms_detections,
        str(OUTPUT_DIR / "04_detections_post_nms.jpg"),
        title=f"Post-NMS: {len(nms_detections)} detections",
    )

    with open(OUTPUT_DIR / "04_nms_detections.json", "w") as f:
        json.dump([
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in nms_detections
        ], f, indent=2)
    save_stage_footer(t)

    # =====================================================================
    # STAGE 5: SAM2 Segmentation
    # =====================================================================
    t = save_stage_header(5, "SAM2 SEGMENTATION")

    from perception.segmentor import Segmentor

    segmentor = Segmentor(device="cuda")
    masks = segmentor.segment(image, nms_detections)

    print(f"  Generated {len(masks)} masks")

    save_masks_visualization(
        image, nms_detections, masks,
        str(OUTPUT_DIR / "05_segmentation_masks.jpg"),
    )
    save_stage_footer(t)

    # =====================================================================
    # STAGE 6: Depth Estimation
    # =====================================================================
    t = save_stage_header(6, "DEPTH ESTIMATION (Metric)")

    from perception.depth import DepthEstimator

    depth_est = DepthEstimator(device="cuda")
    depth_map = depth_est.estimate(image)

    print(f"  Depth shape: {depth_map.shape}")
    valid_depth = depth_map[depth_map > 0]
    if len(valid_depth) > 0:
        print(f"  Range: {valid_depth.min():.2f}m – {valid_depth.max():.2f}m")
        print(f"  Median: {np.median(valid_depth):.2f}m")

    save_depth_visualization(
        depth_map, str(OUTPUT_DIR / "06_depth_map.jpg"),
    )

    # Also save raw depth as .npy
    np.save(str(OUTPUT_DIR / "06_depth_map.npy"), depth_map)
    print(f"  Saved: {OUTPUT_DIR / '06_depth_map.npy'}")
    save_stage_footer(t)

    # =====================================================================
    # STAGE 7: 3D Back-Projection + Calibration + Spatial Relations
    # =====================================================================
    t = save_stage_header(7, "FUSION (3D Back-Projection + Calibration)")

    from fusion.backproject import backproject_to_3d
    from fusion.calibration import apply_scale, auto_calibrate_scale, estimate_intrinsics
    from fusion.spatial_relations import compute_spatial_relations

    intrinsics = estimate_intrinsics(width, height)
    print(f"  Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    objects_3d = []
    for det, mask in zip(nms_detections, masks, strict=False):
        obj = backproject_to_3d(
            mask=mask, depth_map=depth_map, intrinsics=intrinsics,
            label=det.label, confidence=det.confidence,
        )
        if obj is not None:
            objects_3d.append((det, obj))

    print(f"  Back-projected {len(objects_3d)} / {len(nms_detections)} objects to 3D")

    all_obj3d = [obj for _, obj in objects_3d]
    scale_factor = auto_calibrate_scale(all_obj3d)
    print(f"  Scale factor: {scale_factor:.3f}")

    if scale_factor != 1.0:
        apply_scale(all_obj3d, scale_factor)

    spatial_rels = compute_spatial_relations(all_obj3d)
    print(f"  Spatial relations: {len(spatial_rels)}")

    save_stage_footer(t)

    # =====================================================================
    # STAGE 8: Scene Graph Assembly
    # =====================================================================
    t = save_stage_header(8, "SCENE GRAPH ASSEMBLY")

    from fusion.calibration import KNOWN_SIZES
    from scene.models import CalibrationInfo, SceneGraph, SceneObject, SceneRelation

    reference_object = None
    for obj in all_obj3d:
        label_lower = obj.label.lower()
        if any(k in label_lower or label_lower in k for k in KNOWN_SIZES):
            reference_object = obj.label
            break

    scene_objects = []
    for idx, (det, obj) in enumerate(objects_3d):
        scene_objects.append(SceneObject(
            id=idx, label=obj.label, confidence=obj.confidence,
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
        ))

    scene_relations = []
    for rel in spatial_rels:
        scene_relations.append(SceneRelation(
            subject_id=rel.subject_idx, subject_label=rel.subject,
            predicate=rel.predicate,
            object_id=rel.object_idx, object_label=rel.object_label,
            distance_m=round(rel.distance_m, 4),
        ))

    calibration = CalibrationInfo(
        fov_degrees=70.0, scale_factor=round(scale_factor, 4),
        reference_object=reference_object,
        image_width=width, image_height=height,
    )

    scene_graph = SceneGraph(
        objects=scene_objects, relations=scene_relations, calibration=calibration,
    )

    # Save scene graph as text
    scene_text = scene_graph.to_prompt_text()
    print(f"\n{scene_text}")

    with open(OUTPUT_DIR / "08_scene_graph.txt", "w") as f:
        f.write(scene_text)
    print(f"  Saved: {OUTPUT_DIR / '08_scene_graph.txt'}")

    # Save scene graph as JSON
    with open(OUTPUT_DIR / "08_scene_graph.json", "w") as f:
        f.write(scene_graph.model_dump_json(indent=2))
    print(f"  Saved: {OUTPUT_DIR / '08_scene_graph.json'}")

    save_stage_footer(t)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Objects: {len(scene_objects)}")
    print(f"  Relations: {len(scene_relations)}")
    print(f"  Scale: {scale_factor:.3f}")
    if reference_object:
        print(f"  Reference: {reference_object}")
    print(f"\n  All outputs in: {OUTPUT_DIR.resolve()}/")
    print(f"    01_ram_raw_tags.json         — RAM++ raw tags")
    print(f"    02_filtered_tags.json         — Claude-filtered tags")
    print(f"    03_per_tag_detections.json    — per-tag DINO results")
    print(f"    03_detections_pre_nms.jpg     — all detections before NMS")
    print(f"    04_detections_post_nms.jpg    — final detections after NMS")
    print(f"    04_nms_detections.json        — NMS detections data")
    print(f"    05_segmentation_masks.jpg     — SAM2 mask overlay")
    print(f"    06_depth_map.jpg              — colorized depth heatmap")
    print(f"    06_depth_map.npy              — raw depth array")
    print(f"    08_scene_graph.txt            — human-readable scene graph")
    print(f"    08_scene_graph.json           — full scene graph JSON")


if __name__ == "__main__":
    main()
