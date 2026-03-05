from data.data_object import DataObject
from evaluation.evaluation_object import EvaluationObject
from typing import Dict, Any, List, Optional

from model.model_object import ModelObject

_EVAL_REGISTRY = {}


def register_evaluation(name: str):
    """Decorator to register an evaluation metric class by name."""
    def decorator(cls):
        _EVAL_REGISTRY[name] = cls
        return cls
    return decorator


def create_evaluations(metrics_config: List[Dict[str, Any]], 
                       data: DataObject,
                       model: ModelObject) -> List[EvaluationObject]:
    """
    Instantiate all requested evaluation modules from the experiment config.
    
    Args:
        metrics_config: List of dicts, each with "name" and optional "hyperparameters".
        data: The DataObject instance.
        model: The ModelObject instance.
    Returns:
        List of EvaluationObject instances.
    """
    evaluations = []
    for metric in metrics_config:
        name = metric["name"]
        hyperparams = metric.get("hyperparameters", None)
        if name not in _EVAL_REGISTRY:
            raise ValueError(
                f"Evaluation '{name}' is not registered. Available: {list(_EVAL_REGISTRY.keys())}"
            )
        evaluations.append(_EVAL_REGISTRY[name](data, model, hyperparams))
    return evaluations