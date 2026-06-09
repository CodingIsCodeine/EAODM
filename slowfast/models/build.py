"""
Model registry – mimics SlowFast's MODEL_REGISTRY pattern.
"""

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video models.
Registered models can be retrieved by name via MODEL_REGISTRY.get(name)(cfg).
"""


def build_model(cfg):
    """Build a model from the registry using cfg.MODEL.MODEL_NAME."""
    # Ensure all models are registered by importing them
    from slowfast.models import m2mvt  # noqa: F401

    name  = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    return model
