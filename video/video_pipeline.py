"""Orchestrates the full multi-view video pipeline.

video → keyframes → MASt3R reconstruction → per-frame detection →
cross-view merge → scale calibration → spatial relations → SceneGraph
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fusion.backproject import Object3D, backproject_from_pointmap
from fusion.calibration import KNOWN_SIZES, apply_scale, auto_calibrate_scale
from fusion.spatial_relations import compute_spatial_relations
from scene.models import CalibrationInfo, SceneGraph, SceneObject, SceneRelation
from video.keyframes import (
    extract_keyframes_from_video,
    load_keyframes_from_directory,
    save_keyframes_to_temp,
)
from video.mast3r_wrapper import MASt3RReconstructor, MASt3RResult
from video.models import CameraPose, FrameDetection, MergedObject
from video.object_merger import extract_mask_descriptor, merge_objects_across_views

logger = logging.getLogger(__name__)


@dataclass
class VideoPipelineResult:
    """Complete output of the video pipeline."""

    merged_objects: list[MergedObject]
    objects_3d: list[Object3D]
    camera_poses: list[CameraPose]
    mast3r_result: MASt3RResult
    scene_graph: SceneGraph
    keyframe_images: list[Image.Image]
    scale_factor: float
    all_frame_detections: list[list[FrameDetection]] | None = None


class VideoPipeline:
    """Multi-view video pipeline: video → 3D scene graph.

    All ML models are lazy-loaded on first use.
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._mast3r: MASt3RReconstructor | None = None
        self._perception_pipeline = None

    @property
    def mast3r(self) -> MASt3RReconstructor:
        """Lazy-load MASt3R reconstructor."""
        if self._mast3r is None:
            self._mast3r = MASt3RReconstructor(device=self._device)
        return self._mast3r

    @property
    def perception(self):
        """Lazy-load the Phase 1 perception pipeline (detector + segmentor)."""
        if self._perception_pipeline is None:
            from perception.pipeline import PerceptionPipeline

            self._perception_pipeline = PerceptionPipeline(
                device=self._device,
                use_tagger=True,
            )
        return self._perception_pipeline

    @property
    def sam3(self):
        if not hasattr(self, "_sam3") or self._sam3 is None:
            from perception.sam3_detector import Sam3Detector
            self._sam3 = Sam3Detector(device=self._device, confidence_threshold=0.5)
        return self._sam3

    def run(
        self,
        source: str | Path,
        detect: list[str] | None = None,
        use_gemini_tagger: bool = False,
        use_sam3: bool = False,
        every_n_frames: int = 30,
        max_frames: int = 40,
        scene_graph: str = "swin-3",
        add_loop_closure: bool = True,
        keyframe_strategy: str = "uniform",
    ) -> VideoPipelineResult:
        """Run the complete video analysis pipeline.

        Parameters
        ----------
        source:
            Path to a video file (.mp4, .mov) or a directory of images.
        detect:
            Additional object categories to detect.
        use_gemini_tagger:
            Use Gemini 2.5 Flash for tagging instead of RAM++ + Claude.
        use_sam3:
            Use SAM3 for unified detection+segmentation instead of Grounding DINO + SAM2.
        every_n_frames:
            Keyframe extraction interval (for video input).
        max_frames:
            Maximum keyframes to process.
        scene_graph:
            MASt3R pair generation strategy.
        add_loop_closure:
            Add loop-closure pairs for walkthrough videos.

        Returns
        -------
        VideoPipelineResult
        """
        source = Path(source)

        # === Step 1: Extract keyframes ===
        logger.info("Step 1: Extracting keyframes from %s (strategy=%s)", source, keyframe_strategy)
        if source.is_dir():
            keyframes = load_keyframes_from_directory(source, max_frames=max_frames)
        elif source.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
            if keyframe_strategy == "smart":
                from video.smart_keyframes import SmartKeyframeSelector
                selector = SmartKeyframeSelector(device=self._device)
                keyframes = selector.select_keyframes(
                    source, max_frames=max_frames, every_n_frames=every_n_frames,
                )
            else:
                keyframes = extract_keyframes_from_video(
                    source, every_n_frames=every_n_frames, max_frames=max_frames,
                )
        else:
            raise ValueError(
                f"Source must be a video file (.mp4/.mov) or image directory: {source}"
            )

        logger.info("Extracted %d keyframes.", len(keyframes))
        keyframe_images = [img for _, img in keyframes]

        # Save keyframes to temp directory for MASt3R
        cache_dir = tempfile.mkdtemp(prefix="splat_video_")
        image_paths = save_keyframes_to_temp(keyframes, cache_dir)

        # === Step 2: MASt3R reconstruction ===
        logger.info("Step 2: Running MASt3R 3D reconstruction...")
        mast3r_result = self.mast3r.reconstruct(
            image_paths=image_paths,
            scene_graph=scene_graph,
            add_loop_closure=add_loop_closure,
            cache_dir=cache_dir,
        )

        # Build camera poses
        camera_poses = [
            CameraPose(
                frame_idx=i,
                cam2world=mast3r_result.cam2world[i],
                focal_length=mast3r_result.focals[i],
                principal_point=mast3r_result.principal_points[i],
                image_path=image_paths[i],
            )
            for i in range(len(image_paths))
        ]

        # === Step 3: Discover tags from frame 0, apply to all frames ===
        logger.info("Step 3: Discovering object tags from frame 0...")
        tags = self._discover_tags(keyframe_images[0], use_gemini_tagger, detect)
        logger.info("Tags: %s", tags)

        # === Step 4: Per-frame detection + segmentation + back-projection ===
        logger.info("Step 4: Per-frame detection + segmentation...")
        all_frame_detections: list[list[FrameDetection]] = []

        for frame_idx, image in enumerate(keyframe_images):
            logger.info("  Frame %d/%d", frame_idx + 1, len(keyframe_images))

            image_rgb = image.convert("RGB")

            # Detect + Segment
            if use_sam3:
                detections, masks = self.sam3.detect_and_segment(image_rgb, tags)
            else:
                detections = self.perception.detector.detect_per_tag(image_rgb, tags)
                masks = self.perception.segmentor.segment(image_rgb, detections)

            # Get MASt3R maps for this frame
            pointmap = mast3r_result.pts3d[frame_idx]  # (H_m, W_m, 3)
            confmap = mast3r_result.confs[frame_idx]  # (H_m, W_m)
            descmap = mast3r_result.descriptors[frame_idx]  # (H_m, W_m, D)

            # Resize MASt3R maps to match original image resolution
            h_img, w_img = image_rgb.size[1], image_rgb.size[0]
            pointmap_r, confmap_r, descmap_r = self._resize_mast3r_maps(
                pointmap, confmap, descmap, w_img, h_img,
            )

            # Back-project each detection into world frame
            frame_dets: list[FrameDetection] = []
            for det, mask in zip(detections, masks, strict=False):
                obj3d = backproject_from_pointmap(
                    mask=mask,
                    pointmap_world=pointmap_r,
                    confidence_map=confmap_r,
                    label=det.label,
                    confidence=det.confidence,
                )
                if obj3d is None:
                    continue

                descriptor = extract_mask_descriptor(mask, descmap_r)

                frame_dets.append(FrameDetection(
                    frame_idx=frame_idx,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    centroid_world=obj3d.centroid,
                    dimensions_m=obj3d.dimensions_m,
                    points_3d_world=obj3d.points_3d,
                    descriptor=descriptor,
                    mask=mask,
                ))

            all_frame_detections.append(frame_dets)
            logger.info("  Frame %d: %d 3D detections", frame_idx, len(frame_dets))

        # === Step 5: Cross-view object merging ===
        logger.info("Step 5: Merging objects across %d frames...", len(all_frame_detections))
        merged_objects = merge_objects_across_views(all_frame_detections)

        # === Step 5b: Structural surface consolidation ===
        logger.info("Step 5b: Consolidating structural surfaces (wall/floor/ceiling)...")
        from video.structural_merge import merge_structural_surfaces
        merged_objects = merge_structural_surfaces(merged_objects)

        # Convert to Object3D for compatibility with calibration/relations
        objects_3d = [
            Object3D(
                label=mo.label,
                confidence=mo.confidence,
                centroid=mo.centroid_world,
                dimensions_m=mo.dimensions_m,
                distance_m=mo.distance_from_origin,
                points_3d=mo.points_3d_world,
            )
            for mo in merged_objects
        ]

        # === Step 6: Scale calibration (best single view) ===
        logger.info("Step 6: Auto-calibrating scale...")
        scale_factor, ref_object, scale_warning = self._calibrate_scale_best_view(
            all_frame_detections, mast3r_result, keyframe_images, objects_3d,
        )
        if scale_factor != 1.0:
            apply_scale(objects_3d, scale_factor)
            for mo, o3d in zip(merged_objects, objects_3d, strict=False):
                mo.centroid_world = o3d.centroid
                mo.dimensions_m = o3d.dimensions_m
                mo.points_3d_world = o3d.points_3d

        # === Step 7: Spatial relations ===
        logger.info("Step 7: Computing spatial relations...")
        spatial_rels = compute_spatial_relations(objects_3d)

        # === Step 8: Assemble SceneGraph ===
        logger.info("Step 8: Assembling scene graph...")
        scene_objects = [
            SceneObject(
                id=idx,
                label=mo.label,
                confidence=mo.confidence,
                bbox=mo.frame_detections[0].bbox if mo.frame_detections else [0, 0, 0, 0],
                position_m=(
                    round(float(o3d.centroid[0]), 4),
                    round(float(o3d.centroid[1]), 4),
                    round(float(o3d.centroid[2]), 4),
                ),
                dimensions_m=(
                    round(o3d.dimensions_m[0], 4),
                    round(o3d.dimensions_m[1], 4),
                    round(o3d.dimensions_m[2], 4),
                ),
                distance_m=round(o3d.distance_m, 4),
                view_count=mo.view_count,
                coordinate_frame="world",
            )
            for idx, (mo, o3d) in enumerate(zip(merged_objects, objects_3d, strict=False))
        ]

        scene_relations = [
            SceneRelation(
                subject_id=rel.subject_idx,
                subject_label=rel.subject,
                predicate=rel.predicate,
                object_id=rel.object_idx,
                object_label=rel.object_label,
                distance_m=round(rel.distance_m, 4),
            )
            for rel in spatial_rels
        ]

        camera_poses_serializable = [
            {
                "frame_idx": cp.frame_idx,
                "position": cp.position_world.tolist(),
                "focal_length": cp.focal_length,
            }
            for cp in camera_poses
        ]

        calibration = CalibrationInfo(
            fov_degrees=0.0,  # not applicable for multi-view
            intrinsics_source="mast3r",
            scale_factor=round(scale_factor, 4),
            reference_object=ref_object,
            image_width=keyframe_images[0].size[0],
            image_height=keyframe_images[0].size[1],
            coordinate_frame="world",
            num_frames=len(keyframe_images),
            camera_poses=camera_poses_serializable,
            scale_warning=scale_warning,
        )

        scene_graph_obj = SceneGraph(
            objects=scene_objects,
            relations=scene_relations,
            calibration=calibration,
        )

        logger.info(
            "Video pipeline complete: %d frames -> %d merged objects, %d relations",
            len(keyframe_images), len(merged_objects), len(scene_relations),
        )

        return VideoPipelineResult(
            merged_objects=merged_objects,
            objects_3d=objects_3d,
            camera_poses=camera_poses,
            mast3r_result=mast3r_result,
            scene_graph=scene_graph_obj,
            keyframe_images=keyframe_images,
            all_frame_detections=all_frame_detections,
            scale_factor=scale_factor,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_tags(
        self,
        first_frame: Image.Image,
        use_gemini_tagger: bool,
        extra_detect: list[str] | None,
    ) -> list[str]:
        """Discover object tags from the first keyframe."""
        image_rgb = first_frame.convert("RGB")

        if use_gemini_tagger:
            from perception.gemini_tagger import GeminiTagger

            tagger = GeminiTagger()
            tags = tagger.tag(image_rgb)
        else:
            raw_tags = self.perception.tagger.tag(image_rgb)
            tags = self.perception.tag_filter.filter_tags(raw_tags)

        # Merge extra objects
        if extra_detect:
            for obj in extra_detect:
                if obj.lower() not in [t.lower() for t in tags]:
                    tags.append(obj)

        return tags

    def _resize_mast3r_maps(
        self,
        pointmap: np.ndarray,
        confmap: np.ndarray,
        descmap: np.ndarray,
        target_w: int,
        target_h: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize MASt3R maps to match original image resolution."""
        if pointmap.shape[:2] == (target_h, target_w):
            return pointmap, confmap, descmap

        pointmap_r = cv2.resize(pointmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        confmap_r = cv2.resize(confmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        descmap_r = cv2.resize(descmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return pointmap_r, confmap_r, descmap_r

    def _calibrate_scale_best_view(
        self,
        all_frame_detections: list[list[FrameDetection]],
        mast3r_result: MASt3RResult,
        keyframe_images: list[Image.Image],
        objects_3d: list[Object3D],
    ) -> tuple[float, str | None, str | None]:
        """Scale calibration using the best single-view reference object.

        Finds reference objects (door/counter) across all frame detections,
        picks the one with highest confidence + largest mask area, and
        computes scale_factor from that single best measurement.

        Returns
        -------
        tuple[float, str | None, str | None]
            (scale_factor, reference_object_label, scale_warning_or_None)
        """
        from fusion.calibration import _match_known_size

        best_factor: float | None = None
        best_label: str | None = None
        best_score: float = -1.0

        for frame_dets in all_frame_detections:
            for det in frame_dets:
                known_height = _match_known_size(det.label, KNOWN_SIZES)
                if known_height is None:
                    continue

                estimated_height = det.dimensions_m[1]
                if estimated_height <= 0:
                    continue

                # Score: confidence * mask_pixel_count
                mask_area = float(det.mask.sum())
                score = det.confidence * mask_area

                if score > best_score:
                    best_score = score
                    best_factor = known_height / estimated_height
                    best_label = det.label

        if best_factor is not None:
            logger.info(
                "Scale ref (best view): '%s', factor=%.3f (score=%.0f)",
                best_label, best_factor, best_score,
            )
            return best_factor, best_label, None

        # No reference object found
        warning = (
            "No reference object detected — measurements are correct up to scale "
            "(relative proportions accurate, absolute meters may be off)"
        )
        logger.warning(warning)

        # Fall back to merged object calibration (in case merging helped)
        factor = auto_calibrate_scale(objects_3d)
        if factor != 1.0:
            return factor, None, None

        return 1.0, None, warning
