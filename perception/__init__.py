"""Perception module — image to detections, masks, and metric depth."""

from __future__ import annotations

from perception.depth import DepthEstimator
from perception.detector import Detection, ObjectDetector
from perception.pipeline import PerceptionPipeline, PerceptionResult
from perception.segmentor import Segmentor

__all__ = [
    "DepthEstimator",
    "Detection",
    "ObjectDetector",
    "PerceptionPipeline",
    "PerceptionResult",
    "Segmentor",
]
