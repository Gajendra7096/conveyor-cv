"""Classical segmentation methods for Module A."""

from .segmentation import (
    METHOD_NAMES,
    compare_methods,
    labels_to_boxes,
    production_detections,
    segment_frame,
)

__all__ = [
    "METHOD_NAMES",
    "compare_methods",
    "labels_to_boxes",
    "production_detections",
    "segment_frame",
]
