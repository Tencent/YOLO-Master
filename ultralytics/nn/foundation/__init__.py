"""Opt-in, training-only Foundation Teacher interfaces."""

from .projectors import P4AlignmentProjector
from .protocol import FoundationFeatures, FoundationTeacher
from .losses import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss
from .offline import extract_foundation_cache, load_foundation_batch, load_foundation_features, save_foundation_features
from .taps import StudentFeatureTap
from .teachers import (
    DEFAULT_DINOV3_MODEL,
    DEFAULT_SAM3_IMAGE_SIZE,
    DEFAULT_SIGLIP2_MODEL,
    DINOv3Teacher,
    MultiFoundationTeacher,
    SAM3Teacher,
    SigLIP2Teacher,
)
from .routing import (
    FoundationTeacherRouter,
    foundation_multiteacher_summary,
    foundation_teacher_summary,
    routing_kd_loss,
)
from .semantic import (
    RegionSemanticProjector,
    positive_region_pool,
    region_image_loss,
    region_text_loss,
    semantic_distillation_loss,
)

__all__ = [
    "DEFAULT_DINOV3_MODEL",
    "DINOv3Teacher",
    "DEFAULT_SIGLIP2_MODEL",
    "SigLIP2Teacher",
    "DEFAULT_SAM3_IMAGE_SIZE",
    "SAM3Teacher",
    "MultiFoundationTeacher",
    "FoundationFeatures",
    "FoundationTeacher",
    "extract_foundation_cache",
    "load_foundation_batch",
    "load_foundation_features",
    "save_foundation_features",
    "P4AlignmentProjector",
    "StudentFeatureTap",
    "cosine_kd_loss",
    "foreground_token_weights",
    "hybrid_kd_loss",
    "relational_kd_loss",
    "FoundationTeacherRouter",
    "foundation_multiteacher_summary",
    "foundation_teacher_summary",
    "routing_kd_loss",
    "RegionSemanticProjector",
    "positive_region_pool",
    "region_text_loss",
    "region_image_loss",
    "semantic_distillation_loss",
]
