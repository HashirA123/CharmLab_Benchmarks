from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Custom Pytorch Module for Neural Networks
# below is the existing code for the wrapper, but this should be epanded
# to dynamically instantiate different architectures based on the YAML config, and to include methods for loading/saving weights, 
# and for making predictions in the correct format for counterfactual search algorithms.
class PyTorchNeuralNetwork(torch.nn.Module):
    """
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
        params: dict
    ):
        super(PyTorchNeuralNetwork, self).__init__()
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.learning_rate = params['learning_rate']
        self.optimizer = params['optimizer']
        self.loss_function = params['loss_function']
        self.output_activation = params.get('output_activation', None)
        self.device = params['device']

        # Dynamically build the hidden layers based on the provided configuration
        layers = []
        input_size = params['n_inputs']
        for layer_config in params['layers']:
            output_size = layer_config[0]
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())  # Using ReLU activation for hidden layers
            input_size = output_size

        # Output layer
        layers.append(nn.Linear(input_size, params['n_outputs']))

        self.network = nn.Sequential(*layers)

        if self.output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif self.output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)


    # Predictions
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the neural network.

        Returns
        -------
        torch.Tensor: Predicted output tensor (in logit form)

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
                outputs = self.forward(batch_x)
                # pass outputs through the output activation function if specified in the config
                if self.output_activation is not None:
                    outputs = self.output_activation(outputs)

                loss = criterion(outputs, batch_y.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

        return self

    def predict(self, test: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Predicts using the trained neural network.

        Parameters
        -------
        test (torch.Tensor or np.ndarray): Input tensor or array for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        self.eval()
        #y_train_pred = []

        if isinstance(test, np.ndarray):
            test_tensor = torch.tensor(test, dtype=torch.float32).to(self.device)
        else:
            test_tensor = test.to(self.device)

        with torch.no_grad():
            output = self(test_tensor)

            #y_train_pred.extend(output)

        # y_train_pred = torch.stack(y_train_pred)
        return output
