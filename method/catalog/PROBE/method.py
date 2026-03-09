
import pandas as pd
import numpy as np
from typing import Any, Dict, Dict, Optional, Tuple
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
import torch
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.PROBE.library.utils import probe_recourse, perturb_sample
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject
from experiment_utils import deep_merge
import logging


@register_method("PROBE")
class PROBE(MethodObject):
    """
    Implementation of PROBE [1]_.

    .. [1] [1] Martin Pawelczyk,Teresa Datta, Johan Van den Heuvel, Gjergji Kasneci, Himabindu Lakkaraju.2023
            Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse
            https://openreview.net/pdf?id=sC-PmTsiTB(2023).
    """

    def __init__(self, 
                data: DataObject, 
                model: ModelObject, 
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/PROBE/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        
        self._feature_cost = self.config["feature_cost"]
        self._lr = self.config["lr"]
        self._lambda_ = self.config["lambda_"]
        self._n_iter = self.config["n_iter"]
        self._t_max_min = self.config["t_max_min"]
        self._norm = self.config["norm"]
        self._clamp = self.config["clamp"]
        self._loss_type = self.config["loss_type"]
        self._y_target = self.config["y_target"]
        self._binary_cat_features = self.config["binary_cat_features"]
        self._noise_variance = self.config["noise_variance"]
        self._invalidation_target = self.config["invalidation_target"]
        self._inval_target_eps = self.config["inval_target_eps"]

    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        # cat_features_indeces should be a 2d array so that each row corresponds to the indices of the one-hot encoded features for a particular categorical variable.
        cat_features_indices = []
        for features in encoded_feature_names:
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        # So cat_features_indices should look something like [[3,4,5,6]] for the german dataset, 
        # which means the 4 one-hot encoded features of "personal_status_sex" are at those positions 
        # in the encoded dataset. 


        cfs = []
        inval_rates = []
        for index, row in factuals.iterrows():
            
            counterfactual, inval_rate = probe_recourse(
                self._model,
                row.to_numpy().reshape((1, -1)),
                cat_features_indices,
                # binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_cost,
                lr=self._lr,
                lambda_param=self._lambda_,
                y_target=self._y_target,
                n_iter=self._n_iter,
                norm=self._norm,
                t_max_min=self._t_max_min,
                loss_type=self._loss_type,
                clamp=self._clamp,
                invalidation_target=self._invalidation_target,
                inval_target_eps=self._inval_target_eps,
                noise_variance=self._noise_variance
            )
            cfs.append(counterfactual)
            inval_rates.append(inval_rate.item())

        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._data.get_feature_names(expanded=True)) # ensure the feature ordering is correct for the model input
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        print(f"average invalidation rate of generated counterfactuals: {np.mean(inval_rates):.4f}")
        # df_cfs = self._model.get_ordered_features(df_cfs)

        return df_cfs
