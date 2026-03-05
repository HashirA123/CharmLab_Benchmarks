from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from data.data_object import DataObject
from model.model_object import ModelObject


class MethodObject(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    data: data.DataObject
        The data object containing the processed data and metadata.
    model: model.ModelObject
        The model module containing the trained model and its configuration.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
   
    """

    def __init__(self, data: DataObject, model: ModelObject, config_override: Optional[Dict[str, Any]] = None):
        self._data = data
        self._model = model
        self._config_override = config_override

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        pass