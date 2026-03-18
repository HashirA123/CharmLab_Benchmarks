from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim

from data.data_object import DataObject
from model.model_object import ModelObject


class PyTorchLinear(ModelObject, torch.nn.Module):
    """
    PyTorch logistic regression model.

    This model intentionally supports a limited configuration:
    - Single linear layer (no hidden layers)
    - Sigmoid output activation only
    - Binary classification output (`n_output` must be 1)
    """

    def __init__(
        self,
        config_path: str = None,
        data_object: DataObject = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        ModelObject.__init__(self, config_path, data_object, config_override)
        torch.nn.Module.__init__(self)

        self._build_model()

    def _build_model(self) -> None:
        self.batch_size = self._config.get("batch_size", 1000)
        self.epochs = self._config.get("epochs", 1)
        self.learning_rate = self._config.get("learning_rate", 0.001)
        self.optimizer_name = self._config.get("optimizer", "adam").lower()
        self.loss_function = self._config.get("loss_function", "BCE").upper()
        self.activation = self._config.get("output_activation", "sigmoid").lower()
        self.device = self._device

        if self.activation != "sigmoid":
            raise ValueError("PyTorchLinear only supports output_activation='sigmoid'.")

        if int(self._config.get("n_output", 1)) != 1:
            raise ValueError("PyTorchLinear only supports n_output=1 for binary logistic regression.")

        if self.loss_function != "BCE":
            raise ValueError("PyTorchLinear only supports loss_function='BCE'.")

        input_size = len(self._data_object.get_feature_names(expanded=True))
        self.network = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)
        self.fit(self._x_train, self._y_train)

    def _to_feature_array(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x[self.feature_order].to_numpy(dtype=np.float32)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _to_input_tensor(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Tuple[torch.Tensor, bool]:
        if isinstance(x, torch.Tensor):
            return x.to(self.device), True

        x_numeric = self._to_feature_array(x)
        x_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self.device)
        return x_tensor, False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(self, x_train, y_train):
        self.train()

        x_numeric = self._to_feature_array(x_train)
        y_numeric = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        x_train_tensor = torch.tensor(x_numeric, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_numeric, dtype=torch.float32, device=self.device)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "rms":
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(
                "Unsupported optimizer '{}'. Expected one of: adam, sgd, rms.".format(
                    self.optimizer_name
                )
            )

        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def get_train_accuracy(self) -> float:
        x_train = self._to_feature_array(self._x_train)
        predictions = self.predict(x_train)
        accuracy = np.mean(np.asarray(predictions).reshape(-1) == np.asarray(self._y_train).reshape(-1))
        return float(accuracy)

    def get_test_accuracy(self) -> float:
        x_test = self._to_feature_array(self._x_test)
        predictions = self.predict(x_test)
        accuracy = np.mean(np.asarray(predictions).reshape(-1) == np.asarray(self._y_test).reshape(-1))
        return float(accuracy)

    def get_auc(self) -> float:
        x_test = self._to_feature_array(self._x_test)
        y_proba = self.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(np.asarray(self._y_test).reshape(-1), y_proba)
        return float(auc)

    def predict(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_input_tensor(x)

        with torch.no_grad():
            probabilities = self(x_tensor)

        labels = (probabilities > 0.5).float().squeeze(1)

        if is_tensor:
            return labels
        return labels.cpu().numpy()

    def predict_both_classes(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_input_tensor(x)

        with torch.no_grad():
            probabilities = self(x_tensor)

        labels = (probabilities > 0.5).float()
        both_classes_tensor = torch.hstack([1 - labels, labels])

        if is_tensor:
            return both_classes_tensor
        return both_classes_tensor.cpu().numpy()

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self.eval()
        x_tensor, is_tensor = self._to_input_tensor(x)

        with torch.no_grad():
            probabilities = self(x_tensor)

        both_classes_tensor = torch.hstack([1 - probabilities, probabilities])

        if is_tensor:
            return both_classes_tensor
        return both_classes_tensor.cpu().numpy()
