"""Foundation Teacher backend implementations."""

from .dinov2 import DEFAULT_DINOV2_MODEL, DINOv2Teacher
from .dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Teacher
from .multi import MultiFoundationTeacher
from .sam3 import DEFAULT_SAM3_IMAGE_SIZE, SAM3Teacher
from .siglip2 import DEFAULT_SIGLIP2_MODEL, SigLIP2Teacher

__all__ = [
    "DEFAULT_DINOV2_MODEL",
    "DINOv2Teacher",
    "DEFAULT_DINOV3_MODEL",
    "DINOv3Teacher",
    "DEFAULT_SIGLIP2_MODEL",
    "SigLIP2Teacher",
    "DEFAULT_SAM3_IMAGE_SIZE",
    "SAM3Teacher",
    "MultiFoundationTeacher",
]
