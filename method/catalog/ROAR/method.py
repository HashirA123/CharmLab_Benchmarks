
import pandas as pd
import numpy as np
from typing import Any, Dict, Dict, Optional, Tuple
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.ROAR.library.utils import roar_recourse
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject
from experiment_utils import deep_merge, reconstruct_encoding_constraints
import logging


@register_method("ROAR")
class ROAR(MethodObject):
    """
    Implementation of ROAR [1]_.

    .. [1] Upadhyay, S., Joshi, S., & Lakkaraju, H. (2021). Towards Robust and Reliable Algorithmic Recourse. NeurIPS.
    """

    def __init__(self, data: DataObject, 
                model: ModelObject, 
                coeffs: Optional[np.ndarray] = None,
                intercepts: Optional[np.ndarray] = None,
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/ROAR/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        
        self._feature_cost = self.config['feature_cost']
        self._lr = self.config['lr']
        self._lambda_ = self.config['lambda_']
        self._delta_max = self.config['delta_max']
        self._norm = self.config['norm']
        self._t_max_min = self.config['t_max_min']
        self._loss_type = self.config['loss_type']
        self._y_target = self.config['y_target']
        self._binary_cat_features = self.config['binary_cat_features']
        self._loss_threshold = float(self.config['loss_threshold'])
        self._discretize = self.config['discretize']
        self._sample = self.config['sample']
        self._lime_seed = self.config['lime_seed']
        self._enforce_encoding = self.config['enforce_encoding']
        self._seed = self.config['seed']

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
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        # So cat_features_indices should look something like [[3,4,5,6]] for the german dataset, 
        # which means the 4 one-hot encoded features of "personal_status_sex" are at those positions 
        # in the encoded dataset. 

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
                # coeffs, intercepts = self._get_lime_coefficients(factuals)
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

        lime_data = self._model.get_train_data()[0]

        cfs = []
        for index, row in factuals.iterrows():
            if coeffs is None and intercepts is None:
                coeff, intercept = self._get_lime_coefficients(lime_data, row.to_numpy())

            counterfactual = roar_recourse(
                row.to_numpy().reshape((1, -1)),
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
        # TODO: Check counterfactual should be implemented soon.
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 
        # df_cfs = self._model.get_ordered_features(df_cfs)

        return df_cfs

    def _get_lime_coefficients(self, lime_data: np.ndarray, factual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ROAR Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
        we need to find the lime coefficients for every instance.
        """
        np.random.seed(self._lime_seed)

        # coeffs = np.zeros(factuals.shape)
        # intercepts = []

        lime_exp = LimeTabularExplainer(
            training_data = lime_data,
            discretize_continuous = False,
            feature_selection="none", 
            # self._data.encoded_normalized's categorical features contain feature name and value, separated by '_'
            # while self._data.categorical do not contain those additional values.
        )

        # for index, row in factuals.iterrows():
            # factual = row.values
            # print(f"These are the predicted values for the factual instance before passing to lime {self._model.predict_proba(factual)}")
        explanations = lime_exp.explain_instance(
            factual,
            self._model.predict_both_classes, # little misleading from the original, but these predictions have to be labels like [0,1] for positive and [1, 0] for negative.
            num_features=lime_data.shape[1],
            model_regressor=LogisticRegression() 
        )
        intercepts = explanations.intercept[1]
        coefficients = explanations.local_exp[1][0][1]
        # coeffs[index] = coefficients

        return coefficients, np.array(intercepts)