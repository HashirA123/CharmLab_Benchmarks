

import pandas as pd
import torch
from torch import optim
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

@register_method("ClaPROAR")
class ClaPROAR(MethodObject):
    def __init__(self, data: DataObject, model: ModelObject, config_override = None):
        super().__init__(data, model, config_override)


        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/ClaPROAR/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._individual_cost_lambda = self.config['individual_cost_lambda']
        self._external_cost_lambda = self.config['external_cost_lambda']
        self._learning_rate = self.config['learning_rate']
        self._max_iter = self.config['max_iter']
        self._tol = self.config['tol']
        self._target_class = self.config['target_class']

        self._criterion = torch.nn.CrossEntropyLoss()

    
    def compute_yloss(self, x_prime):
        x_prime = x_prime.to(self._device)
        output = self._model.predict_proba(x_prime)
        target_class = torch.tensor(
            [self._target_class] * output.size(0), dtype=torch.long
        ).to(self._device)
        yloss = self._criterion(output, target_class)
        return yloss

    def compute_individual_cost(self, x, x_prime):
        return torch.norm(x - x_prime)

    def compute_external_cost(self, x_prime):
        x_prime = x_prime.to(self._device)
        output = self._model.predict_proba(x_prime)
        target_class = torch.tensor(
            [1 - self._target_class] * output.size(0), dtype=torch.long
        ).to(self._device)
        ext_cost = self._criterion(output, target_class)
        return ext_cost

    def compute_costs(self, x, x_prime):
        yloss = self.compute_yloss(x_prime)
        individual_cost = self.compute_individual_cost(x, x_prime)
        external_cost = self.compute_external_cost(x_prime)

        return (
            yloss
            + self._individual_cost_lambda * individual_cost
            + self._external_cost_lambda * external_cost
        )

    def get_counterfactuals(self, factuals: pd.DataFrame, raw_output: bool = False):
        factuals = factuals[self._data.get_feature_names(expanded=True)] # ensure the feature ordering is correct for the model input

        x = torch.tensor(factuals.values, dtype=torch.float32)

        x_prime = x.clone().detach().requires_grad_(True)
        optimizer_cf = optim.Adam([x_prime], lr=self._learning_rate)

        for i in range(self._max_iter):
            optimizer_cf.zero_grad()

            objective = self.compute_costs(x, x_prime)

            objective.backward()

            optimizer_cf.step()

            if torch.norm(x_prime.grad) < self._tol:
                print(f"Converged at iteration {i+1}")
                break

        cfs = x_prime.detach()
        df_cfs = pd.DataFrame(cfs.numpy(), columns=self._data.get_feature_names(expanded=True))
        if not raw_output:
            df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 
        return df_cfs