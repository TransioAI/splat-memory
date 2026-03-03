"""Perception module — image to detections, masks, and metric depth."""

from __future__ import annotations

from perception.depth import DepthEstimator
from perception.detector import Detection, ObjectDetector
from perception.nms import cross_category_nms
from perception.pipeline import PerceptionPipeline, PerceptionResult
from perception.segmentor import Segmentor
from perception.tag_filter import TagFilter
from perception.tagger import ImageTagger

__all__ = [
    "DepthEstimator",
    "Detection",
    "ImageTagger",
    "ObjectDetector",
    "PerceptionPipeline",
    "PerceptionResult",
    "Segmentor",
    "TagFilter",
    "cross_category_nms",
]
