"""
usage example: python -m experiment --config_path experiments/experiment_config.yml
"""

import argparse
import pandas as pd
import numpy as np
import logging

from config_utils import load_yaml, resolve_layer_config
from data_layer.data_object import DataObject
from model_layer.model_object import ModelObject
from method_layer.method_factory import create_method
from evaluation_layer.evaluation_factory import create_evaluations

# Force registration of all methods and evaluations
import method_layer.ROAR.method  # noqa: F401
import method_layer.PROBE.method  # noqa: F401
import method_layer.RBR.method  # noqa: F401
import method_layer.LARR.method  # noqa: F401
import evaluation_layer.distances  # noqa: F401

_DATA_RAW_PATH = {
    "german": "data_layer/raw_csv/german.csv",
    "compas_carla": "data_layer/raw_csv/compas_carla.csv",
    # add more datasets and their raw data paths here
}

_DATA_CONFIG_PATHS = {
    "german": "data_layer/config_files/data_config_german.yml",
    "compas_carla": "data_layer/config_files/data_config_compas_carla.yml",
    # add more datasets and their config paths here
}

_MODEL_CONFIG_PATHS = {
    "mlp": "model_layer/model_config_mlp.yml",
    # add more model types and their config paths here
}

_METHOD_CONFIG_PATHS = {
    "ROAR": "method_layer/ROAR/library/method_config.yml",
    "PROBE": "method_layer/PROBE/library/method_config.yml",
    "RBR": "method_layer/RBR/library/method_config.yml",
    "LARR": "method_layer/LARR/library/method_config.yml",
    # add more method types and their config paths here
}


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


def run_experiment(config_path: str):
    # load the top level experiment yaml

    exp_config = load_yaml(config_path)
    experiment = exp_config["experiment"]

    setup_logging(experiment.get("logger", "INFO"))

    logger = logging.getLogger("experiment")

    logger.info(f"Running experiment {experiment['name']}")

    # ---------- Data layer loading and config merging -----------
    data_section = exp_config["data"]
    data_config_merged = resolve_layer_config(
        _DATA_CONFIG_PATHS[data_section["name"]],
        data_section.get("overrides")    
    )

    data_object = DataObject(
        data_path=_DATA_RAW_PATH[data_section["name"]],
        config_override=data_config_merged
    )

    logger.info("Data layer loaded and configured.")

    # ---------- Model layer loading and config merging -----------
    model_section = exp_config["model"]
    model_config_merged = resolve_layer_config(
        _MODEL_CONFIG_PATHS[model_section["name"]],
        model_section.get("overrides")
    )

    model_object = ModelObject(
        data_object=data_object,
        config_override=model_config_merged
    )

    logger.info(f"Train accuracy: {model_object.get_train_accuracy():.4f}")
    logger.info(f"Test accuracy:  {model_object.get_test_accuracy():.4f}")

    # ---------- Select factuals for counterfactual generation -----------
    X_test, y_test = model_object.get_test_data()
    factuals = select_factuals(model_object, data_object, X_test, experiment)
    logger.info(f"Selected {len(factuals)} factual instances.")

    # ---------- Method layer loading and config merging -----------
    method_section = exp_config["method"]
    method_config_merged = resolve_layer_config(
        _METHOD_CONFIG_PATHS[method_section["name"]],
        method_section.get("overrides")
    )

    method_object = create_method(
        name=method_section["name"],
        model=model_object,
        data=data_object,
        config_override=method_config_merged
    )

    counterfactuals = method_object.get_counterfactuals(factuals)
    logger.info(f"Generated counterfactuals for {len(counterfactuals)} factual instances.")

    # ---------- Evaluation layer loading and config merging -----------
    evaluation_section = exp_config["evaluation"]
    evaluations = create_evaluations(
        metrics_config=evaluation_section["metrics"],
        data=data_object
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

