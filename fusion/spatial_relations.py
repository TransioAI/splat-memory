"""Compute pairwise spatial relations between scene objects."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .backproject import Object3D

logger = logging.getLogger(__name__)

# Thresholds (meters)
DIRECTIONAL_THRESHOLD = 0.2  # minimum axis difference for directional relations
ON_TOP_HORIZONTAL_THRESHOLD = 0.5  # max XZ distance for on_top_of
NEXT_TO_DISTANCE_THRESHOLD = 1.5  # max Euclidean distance for next_to


@dataclass
class SpatialRelation:
    """A directed spatial relation between two objects."""

    subject_idx: int
    subject: str
    predicate: str
    object_idx: int
    object_label: str
    distance_m: float


def compute_spatial_relations(objects: list[Object3D]) -> list[SpatialRelation]:
    """Compute pairwise spatial relations for all object pairs.

    For every ordered pair (A, B), the following relations are evaluated:
    - left_of / right_of: X-axis comparison
    - above / below: Y-axis comparison (Y-down camera convention)
    - in_front_of / behind: Z-axis comparison
    - on_top_of: A is above B AND horizontally close in XZ plane
    - next_to: Euclidean distance within threshold AND not on_top_of

    Directional relations require a minimum axis difference of 0.2m to avoid
    noisy relations between co-located objects.

    Args:
        objects: List of back-projected Object3D instances.

    Returns:
        List of SpatialRelation describing all meaningful pairwise relations.
    """
    relations: list[SpatialRelation] = []

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue

            delta = a.centroid - b.centroid  # [dx, dy, dz]
            dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
            dist = float(np.linalg.norm(delta))
            horiz_dist = float(np.sqrt(dx**2 + dz**2))

            # --- Directional relations (A relative to B) ---

            # left_of / right_of: compare X coordinates
            if abs(dx) > DIRECTIONAL_THRESHOLD:
                if dx < 0:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="left_of",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="right_of",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))

            # above / below: Y-axis points DOWN, so smaller Y = above
            if abs(dy) > DIRECTIONAL_THRESHOLD:
                if dy < 0:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="above",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="below",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))

            # in_front_of / behind: compare Z (depth) coordinates
            if abs(dz) > DIRECTIONAL_THRESHOLD:
                if dz < 0:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="in_front_of",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_idx=i,
                        subject=a.label,
                        predicate="behind",
                        object_idx=j,
                        object_label=b.label,
                        distance_m=dist,
                    ))

            # --- Composite relations ---

            # on_top_of: A is above B AND horizontally close
            a_is_above_b = dy < -DIRECTIONAL_THRESHOLD
            is_on_top = a_is_above_b and horiz_dist < ON_TOP_HORIZONTAL_THRESHOLD

            if is_on_top:
                relations.append(SpatialRelation(
                    subject_idx=i,
                    subject=a.label,
                    predicate="on_top_of",
                    object_idx=j,
                    object_label=b.label,
                    distance_m=dist,
                ))

            # next_to: close Euclidean distance, but not on_top_of
            if dist < NEXT_TO_DISTANCE_THRESHOLD and not is_on_top:
                relations.append(SpatialRelation(
                    subject_idx=i,
                    subject=a.label,
                    predicate="next_to",
                    object_idx=j,
                    object_label=b.label,
                    distance_m=dist,
                ))

    logger.info("Computed %d spatial relations from %d objects.", len(relations), len(objects))
    return relations
