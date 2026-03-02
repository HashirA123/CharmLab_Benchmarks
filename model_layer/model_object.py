import yaml
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from data_layer.data_object import DataObject
from model_layer.model_builder import PyTorchNeuralNetwork # make use of the existing wrapper class for pytorch models, we can add more wrapper classes for other backends as needed.

class ModelObject:
    """
    A decoupled model instantiation and routing layer.
    
    This module consumes a configuration file and a pre-processed DataModule, 
    acting as a unified interface across different machine learning frameworks. 
    It maintains the strict feature ordering required by the data and provides 
    specialized methods necessary for counterfactual search algorithms.

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

        self._instantiate_model() # Dynamically instantiate the model based on the config
        self._model.to(self._device) # Move model to GPU if available
        
        # get training data from the data object and fit the model
        X_train, X_test, y_train, y_test = self._data_object.get_train_test_split()

        self._x_train = X_train
        self._y_train = y_train
        self._x_test = X_test
        self._y_test = y_test
        
        self._model.fit(self._x_train, self._y_train)


    def _instantiate_model(self) -> None:
        """
        Maps the requested architecture and backend from the YAML config to the 
        corresponding wrapper class (e.g., PyTorchNeuralNetwork, XGBClassifier).
        
        Dynamically fetches input dimensions directly via `self.data_object.get_feature_names(expanded=True)` 
        to ensure the input layer precisely matches the encoded dataset.
        """
        architecture = self._config['architecture']
        backend = self._config['backend']

        params = {
            "n_inputs" : len(self._data_object.get_feature_names(expanded=True)), # Dynamically determine input size
            "n_outputs" : self._config.get('n_output', 2), # Default to 2 for binary classification, can be overridden in config
            "layers" : self._config['hidden_layers'], # describes the number of input and output neurons in each hidden layer, e.g., [[10,100], [100,10]] for two hidden layers with 10 neurons each
            "batch_size" : self._config.get('batch_size', 1000),
            "epochs" : self._config.get('epochs', 1),
            "learning_rate" : self._config.get('learning_rate', 0.001),
            "optimizer" : self._config.get('optimizer', 'adam'),
            "loss_function" : self._config.get('loss_function', 'BCE'),
            "output_activation" : self._config.get('output_activation', 'sigmoid'),
            "device" : self._device
        }

        if backend == 'pytorch' and architecture == 'mlp':
            self._model = PyTorchNeuralNetwork(params)
        else:
            raise NotImplementedError(f"Model architecture '{architecture}' with backend '{backend}' is not implemented yet.")

    # make getters for train and test data
    def get_train_data(self):
        return self._x_train, self._y_train
    

    def get_test_data(self):
        return self._x_test, self._y_test


    def get_train_accuracy(self) -> float:
        """
        Evaluates the model's accuracy on the training set that was set during initialization.
        
        This method serves as a standardized evaluation interface, abstracting away 
        backend-specific evaluation procedures. It ensures that the input features 
        are ordered correctly according to the DataModule's specifications before 
        making predictions and calculating accuracy.
        """
        # ensure X_train is in the correct feature order as specified by the DataObject
        if isinstance(self._x_train, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_train = self._x_train[feature_names].values # reorder columns to match the expected feature order

        predictions = self.predict(x_train)

        accuracy = np.mean(predictions == np.asarray(self._y_train))
        return accuracy


    def get_test_accuracy(self) -> float:
        """
        Evaluates the model's accuracy on the test set that was set during initialization.
        
        This method serves as a standardized evaluation interface, abstracting away 
        backend-specific evaluation procedures. It ensures that the input features 
        are ordered correctly according to the DataModule's specifications before 
        making predictions and calculating accuracy.
        """
        # ensure X_test is in the correct feature order as specified by the DataObject
        if isinstance(self._x_test, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_test = self._x_test[feature_names].values # reorder columns to match the expected feature order

        predictions = self.predict(x_test)
        accuracy = np.mean(predictions == np.asarray(self._y_test))
        return accuracy


    def get_auc(self) -> float:
        """
        Evaluates the model's AUC on the test set that was set during initialization.
        """

        # ensure X_test is in the correct feature order as specified by the DataObject
        if isinstance(self._x_test, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_test = self._x_test[feature_names].values # reorder columns to match the expected feature order

        y_proba = self.predict_proba(x_test)[:, 1] # get probability of positive class
        auc = roc_auc_score(self._y_test, y_proba)
        return auc


    def predict(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Returns raw predictions in the correct format for counterfactual search algorithms.
        
        This method ensures that the input features are ordered according to the 
        DataObject's specifications before passing them to the underlying model. 
        The output is returned in a consistent format (e.g., numpy array or tensor) 
        regardless of the backend.
        """

        # ensure input is in tensor format for PyTorch models, and in the correct feature order as specified by the DataObject
        # should return a list of 1s or 0s.
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x = x[feature_names].values # reorder columns to match the expected feature order
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self._device)
        predictions = self._model.predict(x_tensor)
        if self._config.get('output_activation') == 'sigmoid':
            # If sigmoid is used, convert predictions to binary (0 or 1)
            return (predictions > 0.5).float().cpu().numpy().squeeze() # squeeze to convert from shape (n_samples, 1) to (n_samples,)
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, return the class with the highest probability
            return np.argmax(predictions.cpu().numpy(), axis=1)
        
    
    def predict_both_classes(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Returns the predicted classes for both classes, returns both classes in a numpy array.
        """
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x = x[feature_names].values # reorder columns to match the expected feature order
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self._device)
        predictions = self._model.predict(x_tensor)
        if self._config.get('output_activation') == 'sigmoid':
            # if sigmoid, we need to return both the labels for the classes, which is just the predicted label and its complement (1 - predicted label)
            predicted_labels = (predictions > 0.5).float().cpu().numpy().squeeze() # shape (n_samples,)
            return np.vstack([1 - predicted_labels, predicted_labels]).T # shape (n_samples, 2) with columns [class_0, class_1]
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, the output is already in the form of class probabilities, but we want predicted labels.
            # so we round the probabilities to get the predicted class labels, and then convert back to one-hot encoding format.
            predicted_labels = np.argmax(predictions.cpu().numpy(), axis=1)
            return np.eye(predictions.shape[1])[predicted_labels] # convert to one-hot encoding format
            

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the predicted probabilities for both classes.

        Acts as a universal wrapper that normalizes the output format regardless 
        of whether the underlying backend is Scikit-Learn, PyTorch, or TensorFlow.
        Automatically enforces the correct feature input order before passing data 
        to the underlying model.
        """
        # ensure input is in tensor format for PyTorch models, and in the correct feature order as specified by the DataObject
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x = x[feature_names].values # reorder columns to match the expected feature order
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self._device)
        predictions = self._model.predict(x_tensor)

        if self._config.get('output_activation') == 'sigmoid':
            # if sigmoid, we need to also return the probability of the negative class, which is just 1 - probability of the positive class
            return np.hstack([1 - predictions.cpu().numpy(), predictions.cpu().numpy()]) # shape (n_samples, 2) with columns [prob_class_0, prob_class_1]
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, the output is already in the form of class probabilities, so we can just return it directly.
            return predictions.cpu().numpy()
        