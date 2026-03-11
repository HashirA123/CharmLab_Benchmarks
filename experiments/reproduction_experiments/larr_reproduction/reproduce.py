from copy import deepcopy

import pandas as pd
import numpy as np
import logging

import torch

from experiment_utils import load_yaml, resolve_layer_config, select_factuals, setup_logging
from data.catalog.german.data import GermanData
from model.catalog.mlp.mlp import PyTorchNeuralNetwork

# Force registration of all methods and evaluations
from  method.catalog.LARR.library.utils import LARRecourse, RecourseCost


raw_german = "data/catalog/german/german.csv"
raw_german_corrected = "data/catalog/german/german_corrected.csv"

german_ds_config = "data/catalog/german/data_config_german.yml"
corrected_german_ds_config = "data/catalog/german/data_config_german_corrected.yml"

model_config_path = "model/catalog/mlp/model_config_mlp.yml"

method_config_path = "method/catalog/LARR/library/config.yml"


def run_experiment(config_path: str):
    # load the top level experiment yaml
    torch.manual_seed(0)

    exp_config = load_yaml(config_path)
    experiment = exp_config["experiment"]

    setup_logging(experiment.get("logger", "INFO"))

    logger = logging.getLogger("experiment")

    logger.info(f"Running experiment {experiment['name']}")

    # ---------- Data layer loading and config merging -----------
    data_section = exp_config["data"]

    german_ds_config_merged = resolve_layer_config(german_ds_config, data_section[0].get("overrides"))

    german_object = GermanData(
        data_path=raw_german,
        config_override=german_ds_config_merged
    )

    logger.info("Data layer loaded and configured.")

    # ---------- Model layer loading and config merging -----------

    model_section = exp_config["model"]
    model_config_merged = resolve_layer_config(
        model_config_path,
        model_section.get("overrides")
    )

    current_model = PyTorchNeuralNetwork(
        data_object=german_object,
        config_override=model_config_merged
    )

    # we make the assumtion that the first model is the main one used to 
    # help generate counterfactuals.
    logger.info(f"Test accuracy M1:  {current_model.get_test_accuracy():.4f}")
    logger.info(f"Test AUC M1:  {current_model.get_auc():.4f}")

    # ---------- Select factuals for counterfactual generation -----------
    X_test, _ = current_model.get_test_data()
    X_train, _ = current_model.get_train_data()
    factuals = select_factuals(current_model, german_object, X_test, experiment)
    factuals = factuals.astype(np.float32) # ensure factuals are in numeric format for the methods
    logger.info(f"Selected {len(factuals)} factual instances.")

    # ---------- Method layer loading and config merging -----------
    larr_recourse = LARRecourse(weights=None, bias=None, alpha=0.5)

    experiment2 = experiment.copy()
    experiment2["factual_selection"] = "all"
    recourse_needed_X = select_factuals(current_model, german_object, X_train, experiment2)

    larr_recourse.choose_lambda(
        recourse_needed_X=recourse_needed_X.values.astype(np.float32),
        predict_fn=current_model.predict,
        X_train=X_train.values.astype(np.float32),
        predict_proba_fn=current_model.predict_proba,
        predict_label_fn=current_model.predict_both_classes
    )

    factuals = factuals.to_numpy()

    counter = 0
    running_robustness = 0.0
    running_consistency = 0.0

    for i in range(5):
        instance = factuals[i] 

        J = RecourseCost(instance, larr_recourse.lamb)

        np.random.seed(i)

        weights, bias = larr_recourse.lime_explanation(
            current_model.predict_both_classes,
            X_train.to_numpy().astype(np.float32),
            instance
        )

        print(f"here is weights {weights} and bias {bias} ")

        weights_0, bias_0 = np.round(weights, 4), np.round(bias, 4)

        larr_recourse.weights = weights_0
        larr_recourse.bias = bias_0

        x_r = larr_recourse.get_recourse(instance, beta=1.0)
        weights_r, bias_r = larr_recourse.calc_theta_adv(x_r)
        # theta_r = np.hstack((weights_r, bias_r))
        J_r_opt = J.eval(x_r, weights_r, bias_r)

        print(f"hers is bias_r {bias_r} and weights_r {weights_r} ")
        weights_p = deepcopy(weights_r)
        bias_p = deepcopy(bias_r)
        theta_p = (weights_p, bias_p)

        x_c = larr_recourse.get_recourse(instance, beta=0.0, theta_p=theta_p)
        J_c_opt = J.eval(x_c, *theta_p)

        x = larr_recourse.get_recourse(instance, beta=0.5, theta_p=theta_p)
        weights_r, bias_r = larr_recourse.calc_theta_adv(x)
        # theta_r = np.hstack((weights_r, bias_r))

        J_r = J.eval(x, weights_r, bias_r)
        J_c = J.eval(x, weights_p, bias_p)
        robustness = J_r - J_r_opt
        consistency = J_c - J_c_opt

        running_robustness += robustness
        running_consistency += consistency
        counter += 1
    
    avge_robustness = running_robustness / counter
    avge_consistency = running_consistency / counter
    print(
        f"Avge Robustness: {avge_robustness} and Avge Consistency: {avge_consistency}"
    )  # we should have around 25

    # assert avge_robustness > 0.27 and avge_robustness < 0.29
    # assert avge_consistency > 0.40 and avge_consistency < 0.41

        

if __name__ == "__main__":

    run_experiment("experiments/reproduction_experiments/larr_reproduction/reproduce_larr.yml")

