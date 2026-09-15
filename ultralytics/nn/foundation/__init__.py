"""Opt-in, training-only Foundation Teacher interfaces."""

from .cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_TARGET_SHARD_BYTES,
    FeatureCacheReader,
    FeatureCacheWriter,
    build_cache_key,
    compare_feature_caches,
    verify_feature_cache,
)
from .losses import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss
from .offline import extract_foundation_cache, load_foundation_batch, load_foundation_features, save_foundation_features
from .projectors import P4AlignmentProjector
from .protocol import FoundationFeatures, FoundationTeacher
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

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_DINOV3_MODEL",
    "DEFAULT_SAM3_IMAGE_SIZE",
    "DEFAULT_SIGLIP2_MODEL",
    "DEFAULT_TARGET_SHARD_BYTES",
    "DINOv3Teacher",
    "FeatureCacheReader",
    "FeatureCacheWriter",
    "FoundationFeatures",
    "FoundationTeacher",
    "FoundationTeacherRouter",
    "MultiFoundationTeacher",
    "P4AlignmentProjector",
    "RegionSemanticProjector",
    "SAM3Teacher",
    "SigLIP2Teacher",
    "StudentFeatureTap",
    "build_cache_key",
    "compare_feature_caches",
    "cosine_kd_loss",
    "extract_foundation_cache",
    "foreground_token_weights",
    "foundation_multiteacher_summary",
    "foundation_teacher_summary",
    "hybrid_kd_loss",
    "load_foundation_batch",
    "load_foundation_features",
    "positive_region_pool",
    "region_image_loss",
    "region_text_loss",
    "relational_kd_loss",
    "routing_kd_loss",
    "save_foundation_features",
    "semantic_distillation_loss",
    "verify_feature_cache",
]
