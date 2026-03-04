import yaml
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import torch
from data.data_object import DataObject
from abc import ABC, abstractmethod

class ModelObject(ABC):
    """
    
    This module consumes a configuration file and a pre-processed DataModule, 
    acting as a unified interface across different machine learning frameworks. 
    It maintains the strict feature ordering required by the data and provides 
    specialized methods necessary for counterfactual search algorithms.

    The goal of this module is to provide a consistentstructure for all models to follow,
    specifically in terms of how they instantiated. They must themselves then implement
    the predictions methods and make sure they align with expected formats

    attributes:
    - model: The instantiated machine learning model (e.g., PyTorch, XGBoost).
    - data_module: The injected DataModule instance containing processed data and metadata. 
    - config: The parsed YAML configuration dictionary for model architecture and training hyperparameters.
    """

    def __init__(self, config_path: str = None, data_object: DataObject = None, config_override: Optional[Dict[str, Any]] = None):
        """
        Initializes the ModelObject without redundantly loading raw data.
        
        Args:
            config_path (str): Path to the model configuration YAML.
            data_object (DataObject): The instantiated data layer containing 
                                      the processed data, feature ordering, and bounds.
        """
        self._data_object = data_object
        self._config = yaml.safe_load(open(config_path, 'r')) if config_path is not None else {}
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If a pre-merged config is given, use it entirely (it already contains overrides)
        if config_override is not None:
            self._config = config_override

        # self._instantiate_model() # Dynamically instantiate the model based on the config
        # self._model.to(self._device) # Move model to GPU if available
        
        # get training data from the data object and fit the model
        X_train, X_test, y_train, y_test = self._data_object.get_train_test_split()

        self._x_train = X_train
        self._y_train = y_train
        self._x_test = X_test
        self._y_test = y_test
        
    # make getters for train and test data
    def get_train_data(self):
        return self._x_train, self._y_train
    

    def get_test_data(self):
        return self._x_test, self._y_test

    @abstractmethod
    def get_train_accuracy(self) -> float:
        """
        Evaluates the model's accuracy on the training set that was set during initialization.
        
        This method serves as a standardized evaluation interface, abstracting away 
        backend-specific evaluation procedures. It ensures that the input features 
        are ordered correctly according to the DataModule's specifications before 
        making predictions and calculating accuracy.
        """
        pass

    @abstractmethod
    def get_test_accuracy(self) -> float:
        """
        Evaluates the model's accuracy on the test set that was set during initialization.
        
        This method serves as a standardized evaluation interface, abstracting away 
        backend-specific evaluation procedures. It ensures that the input features 
        are ordered correctly according to the DataModule's specifications before 
        making predictions and calculating accuracy.
        """
        pass

    @abstractmethod
    def get_auc(self) -> float:
        """
        Evaluates the model's AUC on the test set that was set during initialization.
        """
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Returns raw predictions in the correct format for counterfactual search algorithms.
        
        This method ensures that the input features are ordered according to the 
        DataObject's specifications before passing them to the underlying model. 
        The output is returned in a consistent format (e.g., numpy array or tensor) 
        regardless of the backend.
        """
        pass
        
    @abstractmethod
    def predict_both_classes(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Returns the predicted classes for both classes, returns both classes in a numpy array.
        """
        pass
            
    @abstractmethod
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the predicted probabilities for both classes.

        Acts as a universal wrapper that normalizes the output format regardless 
        of whether the underlying backend is Scikit-Learn, PyTorch, or TensorFlow.
        Automatically enforces the correct feature input order before passing data 
        to the underlying model.
        """
        pass
        