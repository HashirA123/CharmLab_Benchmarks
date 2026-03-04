from typing import Optional, Dict, Any

import numpy as np
import yaml

from config_utils import deep_merge
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.RBR.library.utils import rbr_recourse
from method.method_factory import register_method
from model.model_object import ModelObject
from method.method_object import MethodObject
import pandas as pd

@register_method("RBR")
class RBR(MethodObject):
    """
    Implementation of Robust Bayesian Recourse [1]_.

    .. [1] Nguyen, Tuan-Duy Hien, Ngoc Bui, Duy Nguyen, Man-Chung Yue, and Viet Anh Nguyen. 2022.
           "Robust Bayesian Recourse." (UAI 2022)
    """

    def __init__(self, 
                data: DataObject, 
                model: ModelObject, 
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/RBR/library/method_config.yml", 'r'))

        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True)

        self._num_samples = self.config["num_samples"]
        self._perturb_radius = self.config["perturb_radius"]
        self._delta_plus = self.config["delta_plus"]
        self._sigma = self.config["sigma"]
        self._epsilon_op = self.config["epsilon_op"]
        self._epsilon_pe = self.config["epsilon_pe"]
        self._max_iter = self.config["max_iter"]
        self._device = self.config["device"]
        self._clamp = self.config["clamp"]

    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        cat_features_indices = []
        for features in encoded_feature_names:
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.extend(indices)

        x_train, _ = self._model.get_train_data()

        cfs = []

        for idx, row in factuals.iterrows():

            counterfactual = rbr_recourse(
                row.to_numpy().reshape(1, -1), # reshape to 2D array for the model input
                self._model,
                cat_features_indices=cat_features_indices,
                train_data=x_train.to_numpy().astype(np.float32),
                num_samples=self._num_samples,
                perturb_radius=self._perturb_radius,
                delta_plus=self._delta_plus,
                sigma=self._sigma,
                epsilon_op=self._epsilon_op,
                epsilon_pe=self._epsilon_pe,
                max_iter=self._max_iter,
                device=self._device,
                clamp=self._clamp
            )

            cfs.append(counterfactual)

        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._data.get_feature_names(expanded=True)) # ensure the feature ordering is correct for the model input
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 

        return df_cfs