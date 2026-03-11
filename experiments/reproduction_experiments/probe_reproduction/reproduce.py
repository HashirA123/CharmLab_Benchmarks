import logging

import numpy as np
import pandas as pd

from data.catalog.compas.data import CompasData
from evaluation.evaluation_factory import create_evaluations
from experiment_utils import load_yaml, resolve_layer_config, select_factuals, select_factuals, setup_logging
from method.method_factory import create_method
from model.catalog.mlp.mlp import PyTorchNeuralNetwork

# Force registration of all methods and evaluations
import method.catalog.PROBE.method  # noqa: F401
import evaluation.catalog.distances  # noqa: F401
import evaluation.catalog.validity  # noqa: F401


raw_compas_carla = "data/catalog/compas/compas_carla.csv"

compas_carla_ds_config = "data/catalog/compas/data_config_compas_carla.yml"

model_config_path = "model/catalog/mlp/model_config_mlp.yml"

method_config_path = "method/catalog/PROBE/library/config.yml"

def run_experiment(config_path: str):
    # The original experiments dont seem to test against a future model,
    # rather on the same model and test validity and the invalidation rate.

    exp_config = load_yaml(config_path)
    experiment = exp_config["experiment"]

    setup_logging(experiment.get("logger", "INFO"))

    logger = logging.getLogger("experiment")

    # ---------- Data layer loading and config merging -----------
    data_section = exp_config["data"]

    compas_carla_ds_config_merged = resolve_layer_config(compas_carla_ds_config, data_section[0].get("overrides"))

    compas_carla_current_object = CompasData(
        data_path=raw_compas_carla,
        config_override=compas_carla_ds_config_merged
    )

    logger.info("Data layer loaded and configured.")

    # ---------- Model layer loading and config merging -----------

    model_section = exp_config["model"]
    model_config_merged = resolve_layer_config(
        model_config_path,
        model_section.get("overrides")
    )

    current_model = PyTorchNeuralNetwork(
        data_object=compas_carla_current_object,
        config_override=model_config_merged
    )

    # we make the assumtion that the first model is the main one used to 
    # help generate counterfactuals.
    logger.info(f"Test accuracy M1:  {current_model.get_test_accuracy():.4f}")
    logger.info(f"Test AUC M1:  {current_model.get_auc():.4f}")

    # ---------- Select factuals for counterfactual generation -----------
    X_test, y_test = current_model.get_test_data()
    X_train, _ = current_model.get_train_data()
    combined = pd.concat((X_train, X_test), axis=0)
    factuals = select_factuals(current_model, compas_carla_current_object, combined, experiment)
    factuals = factuals.astype(np.float32) # ensure factuals are in numeric format for the methods
    logger.info(f"Selected {len(factuals)} factual instances.")

    # ---------- Method layer loading and config merging -----------
    method_section = exp_config["method"]
    method_config_merged = resolve_layer_config(
        method_config_path,
        method_section.get("overrides")
    )

    method_object = create_method(
        name=method_section["name"],
        model=current_model,  # Assuming the current model object is used for the method
        data=compas_carla_current_object,  # Assuming the current data object is used for the method
        config_override=method_config_merged
    )

    counterfactuals = method_object.get_counterfactuals(factuals)
    logger.info(f"Generated counterfactuals for {len(counterfactuals)} factual instances.")

    # ---------- Evaluation layer loading and config merging -----------
    evaluation_section = exp_config["evaluation"]

    evaluations = create_evaluations(
        metrics_config=evaluation_section["metrics"],
        data=compas_carla_current_object,  # Assuming the current data object is used for evaluation
        model=current_model  # Assuming the current model object is used for evaluation - specifically for future validity.
    )

    results = []
    for eval_module in evaluations:
        eval_result = eval_module.get_evaluation(factuals, counterfactuals)
        results.append(eval_result)
        logger.info(f"Evaluation {eval_module.__class__.__name__} results: {eval_result}")


if __name__ == "__main__":

    run_experiment("experiments/reproduction_experiments/probe_reproduction/reproduce_probe.yml")

