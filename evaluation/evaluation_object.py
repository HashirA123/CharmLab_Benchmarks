from abc import ABC, abstractmethod
import pandas as pd
from data.data_object import DataObject
from model.model_object import ModelObject


class EvaluationObject(ABC):
    def __init__(self, data: DataObject, model: ModelObject, hyperparameters: dict = None):
        """

        Parameters
        ----------
        data: DataObject
            The data object containing the processed data and metadata.
        model: ModelObject
            The model object containing the trained model.
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.data = data 
        self.model = model
        self.hyperparameters = hyperparameters
    
    @abstractmethod
    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute evaluation measure"""
        pass