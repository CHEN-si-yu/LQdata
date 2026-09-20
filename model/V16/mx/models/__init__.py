"""模型实现（import 即注册，见 base.REGISTRY）。"""
from . import gbdt, linear, nn, nn_ref, xgb  # noqa: F401  —— import 触发 @register
from .base import REGISTRY, ModelBase, build, resolve_device  # noqa: F401

__all__ = ["REGISTRY", "ModelBase", "build", "resolve_device"]
