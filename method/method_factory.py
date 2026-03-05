from typing import Any, Dict, Optional

from data.data_object import DataObject
from method.method_object import MethodObject
from model.model_object import ModelObject


_METHOD_REGISTRY = {}


def register_method(name: str):
    """Decorator to register a method class by name."""
    def decorator(cls):
        _METHOD_REGISTRY[name.upper()] = cls
        return cls
    return decorator


def create_method(name: str, 
                  data: DataObject, 
                  model: ModelObject,
                  config_override: Optional[Dict[str, Any]] = None) -> MethodObject:
    """
    Factory function to instantiate a counterfactual method by name.
    
    Args:
        name: The method name (e.g., "ROAR", "PROBE").
        data: The DataObject instance.
        model: The ModelObject instance.
        config_override: Pre-merged method config to inject.
    
    Returns:
        An instance of the requested MethodObject subclass.
    """
    name_upper = name.upper()
    if name_upper not in _METHOD_REGISTRY:
        raise ValueError(
            f"Method '{name}' is not registered. Available: {list(_METHOD_REGISTRY.keys())}"
        )
    method_cls = _METHOD_REGISTRY[name_upper]
    return method_cls(data, model, config_override=config_override)

