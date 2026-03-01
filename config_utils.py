import numpy as np
import torch
import yaml
import copy
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `overrides` into `base`. 
    Values in `overrides` take precedence. Returns a new dict.
    
    Example:
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        overrides = {"b": {"c": 99}, "e": 5}
        result = {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_layer_config(base_config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a layer's base YAML config file and apply any experiment-level overrides.
    
    Args:
        base_config_path: Path to the layer's own config YAML.
        overrides: Dictionary of keys to override from the experiment config.
    
    Returns:
        The merged configuration dictionary.
    """
    base_config = load_yaml(base_config_path)
    if overrides:
        merged = deep_merge(base_config, overrides)
        logger.info(f"Applied {len(overrides)} override(s) to {base_config_path}")
        return merged
    return base_config


def reconstruct_encoding_constraints(instance: torch.Tensor, cat_features_indices: List[int]) -> torch.Tensor:
    """
    For a given instance, ensure that the one-hot encoded categorical features 
    are valid.
    
    Args:
        instance: The input instance as a torch tensor.
        cat_features_indices: A list indicating the indices of the one-hot encoded categorical features.
    
    Returns:
        The instance with reconstructed encoding constraints.
    """
    x_reconstructed = instance.clone()

    for index in cat_features_indices:
        x_reconstructed[index] = torch.clamp(torch.round(x_reconstructed[index]), 0, 1)
    
    return x_reconstructed