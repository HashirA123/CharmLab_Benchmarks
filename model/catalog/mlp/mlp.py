from typing import Any, Dict, Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.data_object import DataObject
from model.model_object import ModelObject

# Custom Pytorch Module for Neural Networks
# below is the existing code for the wrapper, but this should be epanded
# to dynamically instantiate different architectures based on the YAML config, and to include methods for loading/saving weights, 
# and for making predictions in the correct format for counterfactual search algorithms.
class PyTorchNeuralNetwork(ModelObject, torch.nn.Module):
    """
    instantiation of a pytorch nueral net and model object.
    Initializes a PyTorch neural network model with specified number of inputs, outputs, and neurons.

    Parameters
    ----------
    n_inputs (int): Number of input features.
    n_outputs (int): Number of output classes.
    n_neurons (int): Number of neurons in hidden layers.

    Returns
    -------
    PyTorchNeuralNetwork.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(
        self,
        config_path: str = None, 
        data_object: DataObject = None, 
        config_override: Optional[Dict[str, Any]] = None
    ):
        ModelObject.__init__(self, config_path, data_object, config_override)  # Initialize the ModelObject part of the class
        torch.nn.Module.__init__(self)  # Initialize the PyTorch Module
        
        self._build_network()

    def _build_network(self):
        """Builds the neural network architecture based on the provided configuration. """

        self.batch_size = self._config.get('batch_size', 1000)
        self.epochs = self._config.get('epochs', 1)
        self.learning_rate = self._config.get('learning_rate', 0.001)
        self.optimizer = self._config.get('optimizer', 'adam')
        self.loss_function = self._config.get('loss_function', 'BCE')
        self.activation = self._config.get('output_activation', None)
        self.device = self._device

        # Dynamically build the hidden layers based on the provided configuration
        layers = []
        input_size = len(self._data_object.get_feature_names(expanded=True))
        layers.append(nn.Linear(input_size, self._config.get('hidden_layers')[0][0]))  # First hidden layer
        layers.append(nn.ReLU())  # Using ReLU activation for hidden layers

        for layer_config in self._config.get('hidden_layers'):
            if len(layer_config) == 1:
                input_size = layer_config[0]
                output_size = self._config.get('n_output')
                layers.append(nn.Linear(input_size, output_size))
                break  # This is the output layer, so we break after this
            else:
                input_size = layer_config[0]
                output_size = layer_config[1]
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.ReLU())  # Using ReLU activation for hidden layers

        # self.network = nn.Sequential(*layers)

        if self.activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif self.activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)

        layers.append(self.output_activation)  # Add output activation to the end of the network

        self.network = nn.Sequential(*layers)

        # train the model
        self.to(self.device)
        self.fit(self._x_train, self._y_train)


    # Predictions
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the neural network.

        Returns
        -------
        torch.Tensor: Predicted output tensor

        Raises
        -------
        None.
        """
        return self.network(x)

    # Adding extra parameters for training
    def fit(self, x_train, y_train):
        """
        Fits the neural network to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        PyTorchNeuralNetwork: Trained neural network instance.

        Raises
        ------
        None.
        """
        self.train()  # Set the model to training mode
        x_train_tensor = torch.from_numpy(np.array(x_train).astype(np.float32)).to(self.device)
        y_train_tensor = torch.from_numpy(np.array(y_train)).type(torch.LongTensor).to(self.device) 

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # defining the optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer == "rms":
            optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        
        # defining loss function
        if self.loss_function == 'BCE':
            criterion = nn.BCELoss()
        elif self.loss_function == 'MSE':
            criterion = nn.MSELoss()
        
        for _ in range(self.epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_x)
                # pass outputs through the output activation function if specified in the config
                # if self.output_activation is not None:
                #     outputs = self.output_activation(outputs)

                if self.activation == "softmax" and self.loss_function == 'BCE':
                    batch_y = F.one_hot(batch_y, num_classes=2) # convert to one-hot encoding for BCE loss

                # print(f"this is the batch_y {batch_y.unsqueeze(1).float()}")
                # print(f"this is the outputs {outputs}")
                if self.loss_function == 'BCE' and self.activation == "softmax":
                    loss = criterion(outputs, batch_y.float())
                elif self.loss_function == 'BCE' and self.activation == "sigmoid":
                    loss = criterion(outputs, batch_y.unsqueeze(1).float())
                else:
                    loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()

        return self
    
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
        self.eval()  # Set the model to evaluation mode
        is_tensor = False
        # ensure input is in tensor format for PyTorch models, and in the correct feature order as specified by the DataObject
        # should return a list of 1s or 0s.
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_numeric = x[feature_names].to_numpy(dtype=np.float32) # reorder columns to match the expected feature order
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)
        elif isinstance(x, torch.Tensor):
            is_tensor = True
            x_tensor = x.to(self._device)
        else:
            x_numeric = np.array(x, dtype=np.float32) # ensure input is numeric and in numpy array format
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self(x_tensor)

        
        if self._config.get('output_activation') == 'sigmoid':
            # If sigmoid is used, convert predictions to binary (0 or 1)
            if is_tensor:
                return (predictions > 0.5).float()
            else:
                return (predictions > 0.5).float().cpu().numpy().squeeze() # squeeze to convert from shape (n_samples, 1) to (n_samples,)
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, return the class with the highest probability
            if is_tensor:
                return torch.argmax(predictions, dim=1)
            else:
                return np.argmax(predictions.cpu().numpy(), axis=1)
        
    def predict_both_classes(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Returns the predicted classes for both classes, returns both classes in a numpy array.
        """
        self.eval()  # Set the model to evaluation mode
        is_tensor = False
        # ensure input is in tensor format for PyTorch models, and in the correct feature order as specified by the DataObject
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_numeric = x[feature_names].to_numpy(dtype=np.float32) # reorder columns to match the expected feature order
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)
        elif isinstance(x, torch.Tensor):
            is_tensor = True
            x_tensor = x.to(self._device)
        else:
            x_numeric = np.array(x, dtype=np.float32) # ensure input is numeric and in numpy array format
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self(x_tensor)
        
        if self._config.get('output_activation') == 'sigmoid':
            # if sigmoid, we need to return both the labels for the classes, which is just the predicted label and its complement (1 - predicted label)
            predicted_labels = (predictions > 0.5).float().cpu().numpy().squeeze() # shape (n_samples,)
            if is_tensor:
                return torch.stack([1 - predicted_labels, predicted_labels], dim=1) # shape (n_samples, 2) with columns [class_0, class_1]
            else:
                return np.vstack([1 - predicted_labels, predicted_labels]).T # shape (n_samples, 2) with columns [class_0, class_1]
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, the output is already in the form of class probabilities, but we want predicted labels.
            # so we round the probabilities to get the predicted class labels, and then convert back to one-hot encoding format.
            predicted_labels = np.argmax(predictions.cpu().numpy(), axis=1)
            if is_tensor:
                return torch.eye(predictions.shape[1])[predicted_labels] 
            else:
                return np.eye(predictions.shape[1])[predicted_labels] # convert to one-hot encoding format
            
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the predicted probabilities for both classes.

        Acts as a universal wrapper that normalizes the output format regardless 
        of whether the underlying backend is Scikit-Learn, PyTorch, or TensorFlow.
        Automatically enforces the correct feature input order before passing data 
        to the underlying model.
        """
        self.eval()  # Set the model to evaluation mode
        is_tensor = False
        # ensure input is in tensor format for PyTorch models, and in the correct feature order as specified by the DataObject
        if isinstance(x, pd.DataFrame):
            feature_names = self._data_object.get_feature_names(expanded=True)
            x_numeric = x[feature_names].to_numpy(dtype=np.float32) # reorder columns to match the expected feature order
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)
        elif isinstance(x, torch.Tensor):
            is_tensor = True
            x_tensor = x.to(self._device)
        else:
            x_numeric = np.array(x, dtype=np.float32) # ensure input is numeric and in numpy array format
            x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self._device)

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self(x_tensor)

        if self._config.get('output_activation') == 'sigmoid':
            # if sigmoid, we need to also return the probability of the negative class, which is just 1 - probability of the positive class
            if is_tensor:
                return torch.hstack([1 - predictions, predictions]) # shape (n_samples, 2) with columns [prob_class_0, prob_class_1]
            else:
                return np.hstack([1 - predictions.cpu().numpy(), predictions.cpu().numpy()]) # shape (n_samples, 2) with columns [prob_class_0, prob_class_1]
        elif self._config.get('output_activation') == 'softmax':
            # If softmax is used, the output is already in the form of class probabilities, so we can just return it directly.
            if is_tensor:
                return predictions
            else:
                return predictions.cpu().numpy()

