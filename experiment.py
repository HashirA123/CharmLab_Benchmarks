"""
usage example: python -m experiment --config_path experiments/simple_experiments/experiment_config.yml
"""
import argparse
import pandas as pd
import numpy as np
import logging

from experiment_utils import load_yaml, resolve_layer_config, select_factuals, setup_logging
from data.data_object import DataObject
from model.catalog.mlp.mlp_builder import PyTorchNeuralNetwork
from method.method_factory import create_method
from evaluation.evaluation_factory import create_evaluations

# Force registration of all methods and evaluations
import method.catalog.ROAR.method  # noqa: F401
import method.catalog.PROBE.method  # noqa: F401
import method.catalog.RBR.method  # noqa: F401
import method.catalog.LARR.method  # noqa: F401
import method.catalog.WACHTER.method  # noqa: F401
import method.catalog.GS.method  # noqa: F401
import evaluation.catalog.distances  # noqa: F401
import evaluation.catalog.validity  # noqa: F401

_DATA_RAW_PATH = {
    "german": "data/catalog/german/german.csv",
    "german_corrected": "data/catalog/german/german_corrected.csv",
    "compas_carla": "data/catalog/compas/compas_carla.csv",
    # add more datasets and their raw data paths here
}

_DATA_CONFIG_PATHS = {
    "german": "data/catalog/german/data_config_german.yml",
    "german_corrected": "data/catalog/german/data_config_german_corrected.yml",
    "compas_carla": "data/catalog/compas/data_config_compas_carla.yml",
    # add more datasets and their config paths here
}

_MODEL_CONFIG_PATHS = {
    "mlp": "model/catalog/mlp/model_config_mlp.yml",
    # add more model types and their config paths here
}

_METHOD_CONFIG_PATHS = {
    "ROAR": "method/catalog/ROAR/library/config.yml",
    "PROBE": "method/catalog/PROBE/library/config.yml",
    "RBR": "method/catalog/RBR/library/config.yml",
    "LARR": "method/catalog/LARR/library/config.yml",
    "WACHTER": "method/catalog/WACHTER/library/config.yml",
    "GROWING_SPHERES": "method/catalog/GS/library/config.yml",
    # add more method types and their config paths here
}


def run_experiment(config_path: str):
    # load the top level experiment yaml

    exp_config = load_yaml(config_path)
    experiment = exp_config["experiment"]

    setup_logging(experiment.get("logger", "INFO"))

    logger = logging.getLogger("experiment")

    logger.info(f"Running experiment {experiment['name']}")

    # ---------- Data layer loading and config merging -----------
    data_section = exp_config["data"]
    # expand this below to be a list of multiple congifig paths and data objects.
    # this is useful mostly for the evaluation layer, where some metrics might require multiple datasets (e.g. future validity).
    data_configs_merged = []
    for data in data_section:
        data_configs_merged.append(resolve_layer_config(
            _DATA_CONFIG_PATHS[data["name"]],
            data.get("overrides")
        ))

    data_objects = []
    for i, data in enumerate(data_section):
        data_objects.append(DataObject(
            data_path=_DATA_RAW_PATH[data["name"]],
            config_override=data_configs_merged[i]
        ))

    logger.info("Data layer loaded and configured.")

    # ---------- Model layer loading and config merging -----------
    # If we had multiple data objects, then we should create a
    # model object for each of them. The individual models will have the same structure
    # but trained on different data (useful for future validity).
    model_section = exp_config["model"]
    model_config_merged = resolve_layer_config(
        _MODEL_CONFIG_PATHS[model_section["name"]],
        model_section.get("overrides")
    )

    model_objects = []
    for data_obj in data_objects:
        # Since we likely wont have too many different kinds of models,
        # I wont make use of a factory pattern, just use a simple loop and if statements.
        if model_section["name"] == "mlp":
            model_objects.append(PyTorchNeuralNetwork(
                data_object=data_obj,
                config_override=model_config_merged
            ))
        else:
            raise ValueError(f"Unknown model type {model_section['name']}")

    # we make the assumtion that the first model is the main one used to 
    # help generate counterfactuals.
    # logger.info(f"Train accuracy M1: {model_objects[0].get_train_accuracy():.4f}")
    logger.info(f"Test accuracy M1:  {model_objects[0].get_test_accuracy():.4f}")
    logger.info(f"Test AUC M1:  {model_objects[0].get_auc():.4f}")

    if len(model_objects) > 1:
        # logger.info(f"Train accuracy M2: {model_objects[1].get_train_accuracy():.4f}")
        logger.info(f"Test accuracy M2:  {model_objects[1].get_test_accuracy():.4f}")
        logger.info(f"Test AUC M2:  {model_objects[1].get_auc():.4f}")

    # ---------- Select factuals for counterfactual generation -----------
    X_test, y_test = model_objects[0].get_test_data()
    factuals = select_factuals(model_objects[0], data_objects[0], X_test, experiment)
    factuals = factuals.astype(np.float32) # ensure factuals are in numeric format for the methods
    logger.info(f"Selected {len(factuals)} factual instances.")

    # ---------- Method layer loading and config merging -----------
    method_section = exp_config["method"]
    method_config_merged = resolve_layer_config(
        _METHOD_CONFIG_PATHS[method_section["name"]],
        method_section.get("overrides")
    )

    method_object = create_method(
        name=method_section["name"],
        model=model_objects[0],  # Assuming the first model object is used for the method
        data=data_objects[0],  # Assuming the first data object is used for the method
        config_override=method_config_merged
    )

    counterfactuals = method_object.get_counterfactuals(factuals)
    logger.info(f"Generated counterfactuals for {len(counterfactuals)} factual instances.")

    # ---------- Evaluation layer loading and config merging -----------
    evaluation_section = exp_config["evaluation"]

    evaluations = create_evaluations(
        metrics_config=evaluation_section["metrics"],
        data=data_objects[0],  # Assuming the first data object is used for evaluation
        model=model_objects[-1]  # Assuming the last model object is used for evaluation - specifically for future validity.
    )

    results = []
    for eval_module in evaluations:
        eval_result = eval_module.get_evaluation(factuals, counterfactuals)
        results.append(eval_result)
        logger.info(f"Evaluation {eval_module.__class__.__name__} results: {eval_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a counterfactual explanation experiment.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the experiment config YAML file.")
    args = parser.parse_args()

    run_experiment(args.config_path)

