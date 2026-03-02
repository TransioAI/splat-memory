"""Metric depth estimation using Depth Anything V2 (METRIC variant)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Estimate metric depth using Depth Anything V2 Metric Indoor Large.

    **CRITICAL**: This uses the *metric* variant
    ``depth-anything/Depth-Anything-V2-Metric-Indoor-Large`` — **not** the
    relative variant — so that depth values are in metres.

    The model is lazy-loaded on the first call to ``estimate()``.
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self._pipe = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the depth-estimation pipeline if not already loaded."""
        if self._pipe is not None:
            return

        from transformers import pipeline

        logger.info("Loading depth estimation model %s …", self.MODEL_ID)
        self._pipe = pipeline(
            "depth-estimation",
            model=self.MODEL_ID,
            device=self.device,
            torch_dtype=torch.float32,
        )
        logger.info("Depth model loaded on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image: PIL.Image.Image) -> np.ndarray:
        """Return a ``(H, W)`` float32 depth map in **metres**.

        The output is resized to match the *input* image dimensions because
        the model's internal resolution may differ.

        Parameters
        ----------
        image:
            An RGB PIL image.

        Returns
        -------
        np.ndarray
            Depth map of shape ``(H, W)`` with dtype ``float32``.  Values
            represent distance from the camera in metres.
        """
        self._ensure_model()

        image = image.convert("RGB")
        original_w, original_h = image.size

        with torch.no_grad():
            result = self._pipe(image)

        # The pipeline returns a dict with key "depth" (a PIL Image) or
        # "predicted_depth" (a Tensor).  Handle both.
        if "predicted_depth" in result:
            depth_tensor = result["predicted_depth"]
            if torch.is_tensor(depth_tensor):
                depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.float32)
            else:
                depth_np = np.asarray(depth_tensor, dtype=np.float32)
        elif "depth" in result:
            depth_pil = result["depth"]
            depth_np = np.asarray(depth_pil, dtype=np.float32)
        else:
            raise RuntimeError(f"Unexpected depth pipeline output keys: {list(result.keys())}")

        # Resize to original image resolution if necessary
        if depth_np.shape[:2] != (original_h, original_w):
            import cv2

            model_shape = depth_np.shape[:2]
            depth_np = cv2.resize(
                depth_np,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )
            logger.debug(
                "Resized depth map from %s to (%d, %d).",
                model_shape,
                original_h,
                original_w,
            )

        logger.info(
            "Depth estimated: shape=%s, range=[%.2f, %.2f] m",
            depth_np.shape,
            float(depth_np.min()),
            float(depth_np.max()),
        )
        return depth_np
