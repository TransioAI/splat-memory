"""MASt3R wrapper for multi-view 3D reconstruction.

MASt3R must be installed via git clone (not pip):
    git clone --recursive https://github.com/naver/mast3r
    cd mast3r && pip install -e .
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MASt3RResult:
    """Output of MASt3R global alignment."""

    pts3d: list[np.ndarray]  # per-frame (H, W, 3) world-coordinate pointmaps
    confs: list[np.ndarray]  # per-frame (H, W) confidence maps
    descriptors: list[np.ndarray]  # per-frame (H, W, D) feature descriptors (D=24)
    cam2world: list[np.ndarray]  # per-frame (4, 4) camera-to-world transforms
    focals: list[float]  # per-frame focal lengths in pixels
    principal_points: list[tuple[float, float]]  # per-frame (cx, cy)
    image_paths: list[str]  # ordered paths used as input


class MASt3RReconstructor:
    """Lazy-loaded MASt3R model for multi-view 3D reconstruction."""

    MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the MASt3R model."""
        if self._model is not None:
            return

        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as exc:
            raise RuntimeError(
                "MASt3R not installed. Install via:\n"
                "  git clone --recursive https://github.com/naver/mast3r\n"
                "  cd mast3r && pip install -e ."
            ) from exc

        logger.info("Loading MASt3R model %s ...", self.MODEL_ID)
        self._model = AsymmetricMASt3R.from_pretrained(self.MODEL_ID).to(self.device)
        self._model.eval()
        logger.info("MASt3R model loaded on %s", self.device)

    def reconstruct(
        self,
        image_paths: list[str],
        scene_graph: str = "swin-3",
        add_loop_closure: bool = True,
        cache_dir: str | None = None,
        niter1: int = 300,
        niter2: int = 300,
        opt_depth: bool = True,
        shared_intrinsics: bool = False,
    ) -> MASt3RResult:
        """Run MASt3R reconstruction on a sequence of images.

        Steps:
        1. Load images via dust3r utilities
        2. Generate pairs (sliding window + optional loop closure)
        3. Run SparseGA global alignment for consistent poses + dense 3D
        4. Extract per-frame pointmaps, descriptors, confidence, poses

        Parameters
        ----------
        image_paths:
            Ordered list of keyframe file paths.
        scene_graph:
            Pair generation strategy (default: "swin-3").
        add_loop_closure:
            Add loop-closure pairs for walkthrough videos.
        cache_dir:
            Directory for MASt3R cache files. Uses tempdir if None.
        niter1, niter2:
            Optimization iterations for coarse and fine stages.
        opt_depth:
            Whether to optimize depth during alignment.
        shared_intrinsics:
            Whether all frames share the same camera intrinsics.

        Returns
        -------
        MASt3RResult
            Globally aligned pointmaps, descriptors, poses, and intrinsics.
        """
        self._ensure_model()

        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.image_pairs import make_pairs

        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="mast3r_cache_")

        # 1. Load images at MASt3R resolution
        imgs = load_images(image_paths, size=512, verbose=False)

        # 2. Generate pairs
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            symmetrize=True,
        )

        # Add loop closure: pair last 2 frames with first 2
        if add_loop_closure and len(image_paths) > 6:
            self._add_loop_closure_pairs(pairs, imgs)

        logger.info(
            "Running MASt3R SparseGA on %d images with %d pairs ...",
            len(image_paths), len(pairs),
        )

        # 3. Run sparse global alignment
        scene = sparse_global_alignment(
            image_paths,
            pairs,
            cache_dir,
            self._model,
            lr1=0.07,
            niter1=niter1,
            lr2=0.01,
            niter2=niter2,
            device=self.device,
            opt_depth=opt_depth,
            shared_intrinsics=shared_intrinsics,
        )

        # 4. Extract results
        pts3d_tensors, _, confs_tensors = scene.get_dense_pts3d(clean_depth=True)

        pts3d = [
            p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
            for p in pts3d_tensors
        ]
        confs = [
            c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
            for c in confs_tensors
        ]

        # Camera poses: cam2world matrices
        im_poses = scene.get_im_poses()
        cam2world = [
            p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)
            for p in im_poses
        ]

        # Focal lengths
        focals_tensor = scene.get_focals()
        focals = [float(f) for f in focals_tensor]

        # Principal points
        pp_tensor = scene.get_principal_points()
        principal_points = [(float(pp[0]), float(pp[1])) for pp in pp_tensor]

        # Descriptors
        descriptors = self._extract_descriptors(scene, len(image_paths))

        logger.info(
            "MASt3R reconstruction complete: %d frames, pointmap shapes: %s",
            len(pts3d),
            [p.shape for p in pts3d],
        )

        return MASt3RResult(
            pts3d=pts3d,
            confs=confs,
            descriptors=descriptors,
            cam2world=cam2world,
            focals=focals,
            principal_points=principal_points,
            image_paths=image_paths,
        )

    def _add_loop_closure_pairs(self, pairs: list, imgs: list) -> None:
        """Add loop closure pairs between first and last frames."""
        n = len(imgs)
        existing = {(p[0]["idx"], p[1]["idx"]) for p in pairs}

        for i in range(min(2, n)):
            for j in range(max(0, n - 2), n):
                if i != j and (i, j) not in existing and (j, i) not in existing:
                    pairs.append((imgs[i], imgs[j]))
                    existing.add((i, j))

    def _extract_descriptors(
        self,
        scene,
        num_frames: int,
    ) -> list[np.ndarray]:
        """Extract per-frame feature descriptors from MASt3R scene.

        MASt3R stores 24-dimensional descriptors per pixel. After global
        alignment, these may be accessible through the scene's internal state.
        Falls back to empty arrays if not accessible.
        """
        try:
            descs = []
            for i in range(num_frames):
                desc = None
                # Try accessing descriptors from various internal attributes
                for attr in ("_desc", "desc", "stacked_descriptors"):
                    stored = getattr(scene, attr, None)
                    if stored is not None and len(stored) > i:
                        desc = stored[i]
                        break

                if desc is not None:
                    if torch.is_tensor(desc):
                        desc = desc.detach().cpu().numpy()
                    descs.append(np.asarray(desc))
                else:
                    # Fallback: zero descriptors (spatial+label merging still works)
                    h, w = 384, 512
                    if hasattr(scene, "imgs") and len(scene.imgs) > i:
                        shape = scene.imgs[i].shape
                        h, w = shape[0] if len(shape) > 0 else h, shape[1] if len(shape) > 1 else w
                    descs.append(np.zeros((h, w, 24), dtype=np.float32))

            return descs
        except Exception:
            logger.warning(
                "Could not extract MASt3R descriptors; falling back to empty descriptors."
            )
            return [np.zeros((384, 512, 24), dtype=np.float32)] * num_frames
