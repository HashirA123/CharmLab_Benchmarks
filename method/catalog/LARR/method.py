
import pandas as pd
import numpy as np
from typing import Any, Dict, Dict, Optional, Tuple
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.LARR.library.utils import LARRecourse
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject
from experiment_utils import deep_merge
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
        self.config = yaml.safe_load(open("method/catalog/LARR/library/config.yml", 'r'))
        
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

        self._method = LARRecourse(
            weights=self._coeffs,
            bias=self._intercepts,
            alpha=self._alpha,
        )

        # search for optimal lamda during initialization, so that we don't have to do it for every instance during counterfactual generation

        X_train, _ = self._model.get_train_data()
        # print(f"This is what the training data looks like before passing to LARR {X_train}")

        predictions = self._model.predict(X_train)
        recourse_needed = X_train.iloc[
            np.where(predictions == 0)
        ]

        # print(f"this is what recourse needed looks like {recourse_needed}")
        if len(recourse_needed) == 0:
            raise ValueError("No recourse needed for any instance in the training data. Please check your model and data.")
        
        self._method.choose_lambda(
            recourse_needed_X=recourse_needed.to_numpy(),
            predict_fn=self._model.predict,
            X_train=X_train.to_numpy(),
            # predict_proba_fn=self._model.predict_proba,
            predict_label_fn=self._model.predict_both_classes
        )

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
            cat_features_indices.append(indices)

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

            counterfactual = self._method.larr_recourse( # special case here, just to try and keep consistent with original code.
                row.to_numpy().reshape((1, -1)),
                coeff,
                intercept,
                beta=self._beta,
                cat_features_indices=cat_features_indices,
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
            discretize_continuous = False,
            feature_selection="none", 
            # self._data.encoded_normalized's categorical features contain feature name and value, separated by '_'
            # while self._data.categorical do not contain those additional values.
        )

        for index, row in factuals.iterrows():
            factual = row.values
            # print(f"These are the predicted values for the factual instance before passing to lime {self._model.predict_proba(factual)}")
            explanations = lime_exp.explain_instance(
                factual,
                self._model.predict_both_classes, # little misleading from the original, but these predictions have to be labels like [0,1] for positive and [1, 0] for negative.
                num_features=len(self._data.get_feature_names(expanded=True)),
                model_regressor=LogisticRegression() 
            )
            intercepts.append(explanations.intercept[1])
            coefficients = explanations.local_exp[1][0][1]
            coeffs[index] = coefficients


        return coeffs, np.array(intercepts)