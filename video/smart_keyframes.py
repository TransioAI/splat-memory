"""Smart keyframe selection using DINOv2 embeddings and blur detection.

Two-pass adaptive selection:
  Pass 1 — Quality gate: reject blurry and poorly-exposed frames
  Pass 2 — Greedy diversity selection with overlap constraint using DINOv2 cosine distance
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Quality thresholds
BLUR_THRESHOLD = 50.0  # Laplacian variance below this = blurry
DARK_THRESHOLD = 30  # mean pixel intensity below this = too dark
BRIGHT_THRESHOLD = 240  # mean pixel intensity above this = blown out

# DINOv2 selection thresholds (cosine distance = 1 - cosine_similarity)
MIN_DISTANCE = 0.08  # must be at least this different from last selected (skip redundant)
MAX_DISTANCE = 0.75  # must be at most this different (ensure MASt3R overlap)

# DINOv2 model config
DINOV2_MODEL = "dinov2_vits14"  # ViT-S/14: 21M params, fast, good enough for selection
THUMBNAIL_SIZE = 224  # DINOv2 native input size (14x14 patches of 16px = 224)

# Candidate sampling
CANDIDATE_EVERY_N = 5  # check every 5th frame for quality (3x denser than typical uniform)


def _compute_blur_score(frame_bgr: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_brightness(frame_bgr: np.ndarray) -> float:
    """Mean pixel intensity."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


class SmartKeyframeSelector:
    """DINOv2-based keyframe selector with blur rejection."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._transform = transforms.Compose([
            transforms.Resize(THUMBNAIL_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(THUMBNAIL_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _ensure_model(self):
        if self._model is not None:
            return
        logger.info("Loading DINOv2 (%s) for keyframe selection...", DINOV2_MODEL)
        self._model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL, verbose=False)
        self._model = self._model.to(self.device).eval()
        logger.info("DINOv2 loaded on %s", self.device)

    @torch.no_grad()
    def _embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Compute DINOv2 CLS embeddings for a batch of PIL images.

        Returns (N, D) L2-normalized embeddings.
        """
        self._ensure_model()
        tensors = torch.stack([self._transform(img) for img in images]).to(self.device)

        # Process in chunks to avoid OOM on large batches
        chunk_size = 64
        embeddings = []
        for i in range(0, len(tensors), chunk_size):
            chunk = tensors[i:i + chunk_size]
            emb = self._model(chunk)  # (B, D)
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def select_keyframes(
        self,
        video_path: str | Path,
        max_frames: int = 40,
        every_n_frames: int = 30,
        blur_threshold: float = BLUR_THRESHOLD,
        min_distance: float = MIN_DISTANCE,
        max_distance: float = MAX_DISTANCE,
    ) -> list[tuple[int, Image.Image]]:
        """Select keyframes using DINOv2 diversity + blur rejection.

        Parameters
        ----------
        video_path:
            Path to video file.
        max_frames:
            Maximum keyframes to return.
        every_n_frames:
            Used as a secondary cap — candidate sampling is denser (every 5 frames).
        blur_threshold:
            Laplacian variance threshold. Frames below this are rejected.
        min_distance:
            Minimum cosine distance from last selected frame (reject redundant).
        max_distance:
            Maximum cosine distance from last selected frame (ensure overlap).

        Returns
        -------
        list[tuple[int, Image.Image]]
            Selected (frame_index, PIL_image) pairs.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Smart keyframe selection: %s — %d frames, %.1f fps",
            video_path.name, total_frames, fps,
        )

        # === Pass 1: Collect candidates with quality scores ===
        candidates: list[tuple[int, np.ndarray, float]] = []  # (frame_idx, bgr_frame, blur_score)
        frame_idx = 0
        rejected_blur = 0
        rejected_exposure = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % CANDIDATE_EVERY_N == 0:
                blur = _compute_blur_score(frame)
                brightness = _compute_brightness(frame)

                if blur < blur_threshold:
                    rejected_blur += 1
                elif brightness < DARK_THRESHOLD or brightness > BRIGHT_THRESHOLD:
                    rejected_exposure += 1
                else:
                    candidates.append((frame_idx, frame, blur))

            frame_idx += 1

        cap.release()
        logger.info(
            "Pass 1: %d candidates from %d checked (rejected %d blur, %d exposure)",
            len(candidates), total_frames // CANDIDATE_EVERY_N, rejected_blur, rejected_exposure,
        )

        if not candidates:
            logger.warning("No quality frames found! Falling back to uniform sampling.")
            from video.keyframes import extract_keyframes_from_video
            return extract_keyframes_from_video(video_path, every_n_frames, max_frames)

        # Convert to PIL for DINOv2
        pil_images = []
        for _, bgr, _ in candidates:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        # === Embed all candidates in one batch ===
        logger.info("Computing DINOv2 embeddings for %d candidates...", len(candidates))
        embeddings = self._embed_batch(pil_images)  # (N, D) normalized
        logger.info("Embeddings computed: shape %s", embeddings.shape)

        # === Pass 2: Greedy diversity selection ===
        selected_indices: list[int] = [0]  # always start with first candidate
        last_emb = embeddings[0]

        for i in range(1, len(candidates)):
            if len(selected_indices) >= max_frames:
                break

            # Cosine distance from last selected
            cos_sim = float(np.dot(last_emb, embeddings[i]))
            cos_dist = 1.0 - cos_sim

            if cos_dist < min_distance:
                # Too similar to last selected — skip
                continue

            if cos_dist > max_distance:
                # Too different — MASt3R may struggle, but still select
                # (better to have a frame than a gap)
                logger.debug(
                    "Frame %d: high distance %.3f from last selected (may affect MASt3R)",
                    candidates[i][0], cos_dist,
                )

            selected_indices.append(i)
            last_emb = embeddings[i]

        # If we got too few frames, relax min_distance and retry
        if len(selected_indices) < min(max_frames, 5) and len(candidates) > len(selected_indices):
            logger.info(
                "Only %d frames selected, relaxing min_distance from %.3f to %.3f",
                len(selected_indices), min_distance, min_distance * 0.5,
            )
            selected_indices = [0]
            last_emb = embeddings[0]
            relaxed_min = min_distance * 0.5

            for i in range(1, len(candidates)):
                if len(selected_indices) >= max_frames:
                    break
                cos_dist = 1.0 - float(np.dot(last_emb, embeddings[i]))
                if cos_dist < relaxed_min:
                    continue
                selected_indices.append(i)
                last_emb = embeddings[i]

        # Build result
        keyframes: list[tuple[int, Image.Image]] = []
        for idx in selected_indices:
            frame_idx_orig = candidates[idx][0]
            keyframes.append((frame_idx_orig, pil_images[idx]))

        logger.info(
            "Pass 2: Selected %d keyframes from %d candidates (min_dist=%.3f, max_dist=%.3f)",
            len(keyframes), len(candidates), min_distance, max_distance,
        )

        # Log frame indices for debugging
        frame_indices = [kf[0] for kf in keyframes]
        logger.info("Selected frame indices: %s", frame_indices)

        return keyframes
