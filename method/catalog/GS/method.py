import pandas as pd
import numpy as np
from typing import Any, Dict, Dict, Optional, Tuple
import yaml
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.GS.library.util import growing_spheres_search
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject
from experiment_utils import deep_merge, reconstruct_encoding_constraints
import logging


@register_method("GROWING_SPHERES")
class GrowingSpheres(MethodObject):
    """
    Implementation of Growing Spheres from Laugel et.al. [1]_.

    Notes
    -----
    - Restrictions
        Growing Spheres works at the moment only for data with dropped first column of binary categorical features.

    .. [1] Thibault Laugel, Marie-Jeanne Lesot, Christophe Marsala, Xavier Renard, and Marcin Detyniecki. 2017.
            Inverse Classification for Comparison-based Interpretability in Machine Learning.
            arXiv preprint arXiv:1712.08443(2017).
    """

    def __init__(self, data: DataObject, 
                model: ModelObject, 
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/GS/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        
        # all these should come from the data object
        self._immutables = self._data.get_mutable_features(mutable=False)
        self._mutables = self._data.get_mutable_features(mutable=True)
        self._continuous = self._data.get_continuous_features()
        categorical_enc = self._data.get_categorical_features(expanded=True) # need to flatten this
        self._categorical_enc = []
        for row in categorical_enc:
            self._categorical_enc.extend(row)      
        
        # other attributes
        self._n_search_samples = self.config['n_search_samples']
        self._p_norm = self.config['p_norm']
        self._step = self.config['step']
        self._max_iter = self.config['max_iter']


    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        # divide up keys
        keys_mutable_continuous = list(set(self._mutables) - set(self._categorical_enc))
        keys_mutable_binary = list(set(self._mutables) - set(self._continuous))

        print(f"keys_mutable_continuous {keys_mutable_continuous}")
        print(f"keys_mutable_binary {keys_mutable_binary}")

        cfs = []
        for index, row in factuals.iterrows():

            instance_immutable_replicated = np.repeat(
                row[self._immutables].values.reshape(1, -1), self._n_search_samples, axis=0
            )
            instance_replicated = np.repeat(
                row.values.reshape(1, -1), self._n_search_samples, axis=0
            )
            instance_mutable_replicated_continuous = np.repeat(
                row[keys_mutable_continuous].values.reshape(1, -1),
                self._n_search_samples,
                axis=0,
            )

            counterfactual = growing_spheres_search(
                row.to_numpy().reshape(1, -1),
                self._mutables,
                self._immutables,
                keys_mutable_continuous,
                keys_mutable_binary,
                self._feature_order,
                self._model,
                instance_immutable_replicated,
                instance_replicated,
                instance_mutable_replicated_continuous,
                n_search_samples=self._n_search_samples,
                p_norm=self._p_norm,
                step=self._step,
                max_iter=self._max_iter,
            )
            cfs.append(counterfactual)

        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._feature_order, index=factuals.index)
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 
        return df_cfs