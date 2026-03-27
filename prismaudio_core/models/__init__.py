"""
PrismAudio model modules for inference.

Re-exports create_model_from_config from the factory module.
"""

from prismaudio_core.factory import create_model_from_config

__all__ = ["create_model_from_config"]
