from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from experiment_utils import deep_merge
from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from method.catalog.WACHTER.library.util import wachter_recourse
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

@register_method("WACHTER")
class WACHTER(MethodObject):
    """
    Implmentation of Wachter et al. [1]_.

    .. [1] Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harvard Journal of Law & Technology, 31(2), 841-887.
    """
    
    def __init__(self, data: DataObject, 
                model: ModelObject, 
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/WACHTER/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True) # ensure the feature ordering is correct for the model input
        
        self._feature_cost = self.config['feature_cost']
        self._lr = self.config['lr']
        self._lambda_ = self.config['lambda_']
        self._n_iter = self.config['n_iter']
        self._t_max_min = self.config['t_max_min']
        self._clamp = self.config['clamp']
        self._loss_type = self.config['loss_type']
        self._norm = self.config['norm']
        self._y_target = self.config['y_target']
        self._binary_cat_features = self.config['binary_cat_features']

    
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        factuals = factuals.reset_index()
        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        cat_features_indices = []
        # TODO: this logic of getting categorical feature indices is repeated across multiple methods, should be moved to a utility function in the data object or a shared utility file
        for features in encoded_feature_names:
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        cfs = []
        for index, row in factuals.iterrows():
            
            counterfactual = wachter_recourse(
                x=row.to_numpy().reshape((1, -1)),
                model=self._model,
                cat_feature_indices=cat_features_indices,
                # binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_cost,
                lr=self._lr,
                n_iter=self._n_iter,
                clamp=self._clamp,
                lambda_param=self._lambda_,
                y_target=self._y_target,
                norm=self._norm,
                t_max_min=self._t_max_min,
                loss_type=self._loss_type,
            )
            cfs.append(counterfactual)
        
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._feature_order)
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs

