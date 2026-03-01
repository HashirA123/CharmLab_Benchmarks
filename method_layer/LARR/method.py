
import pandas as pd
import numpy as np
from typing import Any, Dict, Dict, Optional, Tuple
from lime.lime_tabular import LimeTabularExplainer
import yaml
from data_layer.data_object import DataObject
from evaluation_layer.utils import check_counterfactuals
from method_layer.LARR.library.method_utils import larr_recourse
from method_layer.method_factory import register_method
from method_layer.method_object import MethodObject
from model_layer.model_object import ModelObject
from config_utils import deep_merge
import logging


@register_method("LARR")
class LARR(MethodObject):
    """
    Implementation of LARR [1]_.

    .. [1] Kayastha, K., Gkatzelis, V., Jabbari, S. (2025). Learning-Augmented Robust Algorithmic Recourse. Drexel University. (https://arxiv.org/pdf/2410.01580)
    """

    def __init__(self, data: DataObject, 
                model: ModelObject, 
                coeffs: Optional[np.ndarray] = None,
                intercepts: Optional[np.ndarray] = None,
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method_layer/LARR/library/method_config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        
        self._feature_cost = self.config['feature_cost']
        self._alpha = self.config['alpha']
        self._beta = self.config['beta']
        self._loss_type = self.config['loss_type']
        self._lime_seed = self.config['lime_seed']

        self._coeffs = coeffs
        self._intercepts = intercepts

    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        cat_features_indices = []
        for features in encoded_feature_names:
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.extend(indices)

        coeffs = self._coeffs
        intercepts = self._intercepts

        # Calculate coefficients and intercept (if not given) and reshape to match the shape that LIME generates
        # If Model linear then extract coefficients and intercepts from raw model directly
        # If Model mlp then use LIME to generate the coefficients
        if (coeffs is None) or (intercepts is None):
            if self._model._config["architecture"] == "linear":
               raise ValueError("Depreciated support for linear, experiment with mlp instead. If you want to use linear, please provide coefficients and intercepts in the correct shape.")
            elif self._model._config["architecture"] == "mlp":
                logging.info("Start generating LIME coefficients")
                coeffs, intercepts = self._get_lime_coefficients(factuals)
                logging.info("Finished generating LIME coefficients")
            else:
                raise ValueError(
                    f"Model architecture {self._model._config['architecture']} not supported in ROAR recourse method"
                )
        else:
            # Coeffs and intercepts should be numpy arrays of shape (num_features,) and () respectively
            if (len(coeffs.shape) != 1) or (coeffs.shape[0] != factuals.shape[1]):
                raise ValueError(
                    "Incorrect shape of coefficients. Expected shape: (num_features,)"
                )
            if len(intercepts.shape) != 0:
                raise ValueError("Incorrect shape of coefficients. Expected shape: ()")

            # Reshape to desired shape: (num_of_instances, num_of_features)
            coeffs = np.vstack([self._coeffs] * factuals.shape[0])
            intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(
                axis=1
            )

        
        
        cfs = []
        for index, row in factuals.iterrows():
            coeff = coeffs[index]
            intercept = intercepts[index]

            counterfactual = larr_recourse(
                row.to_numpy(), #.reshape((1, -1)),
                coeff,
                intercept,
                cat_features_indices,
                # binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_cost,
                lr=self._lr,
                lambda_param=self._lambda_,
                delta_max=self._delta_max,
                y_target=self._y_target,
                norm=self._norm,
                t_max_min=self._t_max_min,
                loss_type=self._loss_type,
                loss_threshold=self._loss_threshold,
                enforce_encoding=self._enforce_encoding,
                seed=self._seed,
            )
            cfs.append(counterfactual)

        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._data.get_feature_names(expanded=True)) # ensure the feature ordering is correct for the model input
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 

        return df_cfs

    def _get_lime_coefficients(self, factuals: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        LARR Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
        we need to find the lime coefficients for every instance.
        """
        np.random.seed(self._lime_seed)

        coeffs = np.zeros(factuals.shape)
        intercepts = []
        lime_data = self._model.get_train_data()[0] # get the training data features from the model module, which should already be in the correct feature order and format for the model input
        lime_label = self._model.get_train_data()[1] # get the training data labels from the model module

        lime_exp = LimeTabularExplainer(
            training_data = lime_data,
            training_labels = lime_label,
            mode="regression",
            discretize_continuous=False,
            feature_selection="none",
        )

        for index, row in factuals.iterrows():
            factual = row.values
            explanations = lime_exp.explain_instance(
                factual,
                self._model.predict_proba,
                num_features=len(self._data.get_feature_names(expanded=True)),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[index][tpl[0]] = tpl[1]

        return coeffs, np.array(intercepts)