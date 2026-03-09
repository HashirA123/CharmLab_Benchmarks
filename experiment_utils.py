import numpy as np
import pandas as pd
import torch
import yaml
import copy
from typing import Any, Dict, List, Optional
import logging

from data.data_object import DataObject
from model.model_object import ModelObject

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


def reconstruct_encoding_constraints(instance: torch.Tensor, cat_features_indices: List[List[int]]) -> torch.Tensor:
    """
    For a given instance, ensure that the one-hot encoded categorical features 
    are valid.
    
    Args:
        instance: The input instance as a torch tensor.
        cat_features_indices: A list of lists indicating the indices of the one-hot encoded categorical features.
    
    Returns:
        The instance with reconstructed encoding constraints.
    """
    x_reconstructed = instance.clone()

    for features in cat_features_indices:
        if len(features) > 1:
            max_indices_in_group = torch.argmax(x_reconstructed[:, features], dim=1)

            x_reconstructed[:, features] = 0
            row_indices = torch.arange(x_reconstructed.size(0))
            absolute_feature_indices = torch.tensor(features, device=x_reconstructed.device)[max_indices_in_group]
            x_reconstructed[row_indices, absolute_feature_indices] = 1.0
        elif len(features) == 1:
            # For binary categorical features that are one-hot encoded into a single column, we can simply round the value to 0 or 1
            x_reconstructed[:, features[0]] = torch.round(x_reconstructed[:, features[0]])
    
    return x_reconstructed


def setup_logging(name: str):
    level = getattr(logging, name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def select_factuals(model: ModelObject, data: DataObject, X_test, config) -> pd.DataFrame:
    num_factuals = config.get("num_factuals", 5)
    factual_selection = config.get("factual_selection", "negative_class")

    if factual_selection == "negative_class":
        prediction = model.predict(X_test)
        neg_indices = np.where(prediction == 0)[0] # returns the indices
        selected = X_test.to_numpy()[neg_indices][:num_factuals]
    elif factual_selection == "all":
        prediction = model.predict(X_test)
        neg_indices = np.where(prediction == 0)[0] # returns the indices
        selected = X_test.to_numpy()[neg_indices]
    else:
        raise ValueError(f"Unknown factual selection method {factual_selection}")
    
    return pd.DataFrame(selected, columns=data.get_feature_names(expanded=True))