import argparse
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.data_object import DataObject
from experiment_utils import load_yaml, resolve_layer_config, setup_logging
from method.method_factory import create_method
from model.catalog.linear.linear import PyTorchLinear
from model.catalog.mlp.mlp import PyTorchNeuralNetwork

# Force method registration.
import method.catalog.GREEDY.method  # noqa: F401


_DATA_RAW_PATH = {
	"boston_housing": "data/catalog/boston_housing/boston_housing.csv",
}

_DATA_CONFIG_PATHS = {
	"boston_housing": "data/catalog/boston_housing/data_config_boston.yml",
}

_MODEL_CONFIG_PATH = "model/catalog/linear/config.yml"
_METHOD_CONFIG_PATH = "method/catalog/GREEDY/library/config.yml"


class Autoencoder(nn.Module):
	"""Simple feedforward autoencoder for realism scoring."""

	def __init__(self, input_dim: int):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
		)
		self.decoder = nn.Sequential(
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, input_dim),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.decoder(self.encoder(x))


def build_autoencoder(input_dim: int, device: torch.device) -> Autoencoder:
	model = Autoencoder(input_dim)
	model.to(device)
	return model


def train_autoencoder(
	model: Autoencoder,
	data_np: np.ndarray,
	epochs: int,
	batch_size: int,
	learning_rate: float,
	device: torch.device,
) -> None:
	tensor_data = torch.tensor(data_np.astype(np.float32), dtype=torch.float32)
	loader = DataLoader(
		TensorDataset(tensor_data),
		batch_size=batch_size,
		shuffle=True,
	)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.MSELoss()

	model.train()
	for _ in range(epochs):
		for (batch_x,) in loader:
			batch_x = batch_x.to(device)
			optimizer.zero_grad()
			reconstruction = model(batch_x)
			loss = criterion(reconstruction, batch_x)
			loss.backward()
			optimizer.step()


def calculate_standard_deviation(values: List[float]) -> float:
	return float(np.std(values))


def calculate_l1_distance(counterfactuals: pd.DataFrame, factuals: pd.DataFrame) -> float:
	return float(np.mean(np.abs(counterfactuals - factuals).sum(axis=1)))


def _predict_autoencoder(model: Autoencoder, values: np.ndarray, device: torch.device) -> np.ndarray:
	model.eval()
	with torch.no_grad():
		x = torch.tensor(values.astype(np.float32), dtype=torch.float32, device=device)
		reconstruction = model(x)
	return reconstruction.cpu().numpy()


def calculate_realism_score(
	counterfactuals: pd.DataFrame,
	ae_all: Autoencoder,
	ae_target: Autoencoder,
	device: torch.device,
) -> float:
	counterfactual_array = counterfactuals.to_numpy(dtype=np.float32)

	reconstruction_all = _predict_autoencoder(ae_all, counterfactual_array, device)
	loss_all = np.sum(np.square(counterfactual_array - reconstruction_all), axis=1)

	reconstruction_target = _predict_autoencoder(ae_target, counterfactual_array, device)
	loss_target = np.sum(np.square(counterfactual_array - reconstruction_target), axis=1)

	epsilon = 1e-8
	im1_scores = loss_target / (loss_all + epsilon)
	return float(np.mean(im1_scores))


def _to_numpy_prediction(predictions: Any) -> np.ndarray:
	if isinstance(predictions, torch.Tensor):
		return predictions.detach().cpu().numpy()
	return np.asarray(predictions)


def _select_negative_instances(
	model: PyTorchNeuralNetwork,
	candidates: pd.DataFrame,
	num_factuals: int,
) -> pd.DataFrame:
	predictions = _to_numpy_prediction(model.predict(candidates)).reshape(-1)
	negative_mask = predictions == 0
	return candidates.loc[negative_mask].head(num_factuals)


def _prepare_valid_counterfactuals(
	factuals: pd.DataFrame,
	counterfactuals: pd.DataFrame,
	target_column: str,
	feature_order: List[str],
) -> pd.DataFrame:
	factual_features = factuals[feature_order]
	cf_features = counterfactuals.drop(columns=[target_column], errors="ignore")
	cf_features = cf_features[feature_order]

	valid_mask = ~cf_features.isna().any(axis=1)
	factual_valid = factual_features.loc[valid_mask]
	cf_valid = cf_features.loc[valid_mask]

	if cf_valid.empty:
		raise ValueError("GREEDY did not produce any valid counterfactuals for this dataset.")

	return factual_valid, cf_valid


def _run_on_dataset(
	dataset_name: str,
	dataset_overrides: Dict[str, Any],
	model_overrides: Dict[str, Any],
	method_overrides: Dict[str, Any],
	ae_config: Dict[str, Any],
	num_factuals: int,
	logger: logging.Logger,
) -> Dict[str, float]:
	if dataset_name not in _DATA_RAW_PATH or dataset_name not in _DATA_CONFIG_PATHS:
		raise ValueError(f"Unsupported dataset '{dataset_name}'.")

	data_config = resolve_layer_config(_DATA_CONFIG_PATHS[dataset_name], dataset_overrides)
	data_object = DataObject(
		data_path=_DATA_RAW_PATH[dataset_name],
		config_override=data_config,
	)

	model_config = resolve_layer_config(_MODEL_CONFIG_PATH, model_overrides)
	model = PyTorchLinear(data_object=data_object, config_override=model_config)

	X_test, _ = model.get_test_data()
	factuals = _select_negative_instances(model, X_test, num_factuals)
	if factuals.empty:
		raise ValueError(f"No negative instances found in test data for dataset '{dataset_name}'.")

	method_config = resolve_layer_config(_METHOD_CONFIG_PATH, method_overrides)
	greedy_method = create_method(
		name="GREEDY",
		data=data_object,
		model=model,
		config_override=method_config,
	)
	counterfactuals = greedy_method.get_counterfactuals(factuals.astype(np.float32))

	target_column = data_object.get_target_column()
	feature_order = data_object.get_feature_names(expanded=True)
	factuals_valid, counterfactuals_valid = _prepare_valid_counterfactuals(
		factuals,
		counterfactuals,
		target_column,
		feature_order,
	)

	processed_df = data_object.get_processed_data()
	all_features = processed_df.drop(columns=[target_column]).to_numpy(dtype=np.float32)
	target_features = (
		processed_df[processed_df[target_column] == 1]
		.drop(columns=[target_column])
		.to_numpy(dtype=np.float32)
	)

	if target_features.shape[0] == 0:
		raise ValueError(f"No positive-class samples found for dataset '{dataset_name}'.")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	input_dim = all_features.shape[1]

	ae_all = build_autoencoder(input_dim, device)
	ae_target = build_autoencoder(input_dim, device)

	train_autoencoder(
		ae_all,
		all_features,
		epochs=int(ae_config["epochs"]),
		batch_size=int(ae_config["batch_size"]),
		learning_rate=float(ae_config["learning_rate"]),
		device=device,
	)
	train_autoencoder(
		ae_target,
		target_features,
		epochs=int(ae_config["epochs"]),
		batch_size=int(ae_config["batch_size"]),
		learning_rate=float(ae_config["learning_rate"]),
		device=device,
	)

	l1_distance = calculate_l1_distance(counterfactuals_valid, factuals_valid)
	realism_score = calculate_realism_score(counterfactuals_valid, ae_all, ae_target, device)
	valid_ratio = float(len(counterfactuals_valid) / len(factuals))

	logger.info(
		"Dataset=%s | factuals=%d | valid_cfs=%d | valid_ratio=%.3f | l1=%.6f | im1=%.6f",
		dataset_name,
		len(factuals),
		len(counterfactuals_valid),
		valid_ratio,
		l1_distance,
		realism_score,
	)

	return {
		"dataset": dataset_name,
		"l1_distance": l1_distance,
		"realism_score": realism_score,
		"valid_ratio": valid_ratio,
	}


def run_reproduction(config_path: str) -> List[Dict[str, float]]:
	config = load_yaml(config_path)
	experiment = config["experiment"]
	data_section = config["data"]
	model_section = config["model"]
	method_section = config["method"]

	setup_logging(experiment.get("logger", "info"))
	logger = logging.getLogger("greedy_reproduction")

	seed = int(experiment.get("seed", 42))
	np.random.seed(seed)
	torch.manual_seed(seed)

	ae_config = config.get(
		"autoencoder",
		{"epochs": 50, "batch_size": 32, "learning_rate": 1e-3},
	)

	num_factuals = int(experiment.get("num_factuals", 5))

	results = []
	for dataset_cfg in data_section:
		result = _run_on_dataset(
			dataset_name=dataset_cfg["name"],
			dataset_overrides=dataset_cfg.get("overrides", {}),
			model_overrides=model_section.get("overrides", {}),
			method_overrides=method_section.get("overrides", {}),
			ae_config=ae_config,
			num_factuals=num_factuals,
			logger=logger,
		)
		results.append(result)

	l1_distances = [result["l1_distance"] for result in results]
	realism_scores = [result["realism_score"] for result in results]
	valid_ratios = [result["valid_ratio"] for result in results]

	l1_std = calculate_standard_deviation(l1_distances)
	realism_std = calculate_standard_deviation(realism_scores)
	min_valid_ratio = float(min(valid_ratios))

	l1_std_threshold = float(config.get("assertions", {}).get("l1_std_max", 0.1))
	realism_std_threshold = float(config.get("assertions", {}).get("realism_std_max", 0.1))
	valid_ratio_threshold = float(config.get("assertions", {}).get("min_valid_ratio", 0.5))

	logger.info("L1 std across datasets: %.6f", l1_std)
	logger.info("IM1 std across datasets: %.6f", realism_std)
	logger.info("Minimum valid counterfactual ratio across datasets: %.6f", min_valid_ratio)

	assert l1_std < l1_std_threshold, (
		f"L1 std {l1_std:.6f} exceeds threshold {l1_std_threshold:.6f}"
	)
	assert realism_std < realism_std_threshold, (
		f"IM1 std {realism_std:.6f} exceeds threshold {realism_std_threshold:.6f}"
	)
	assert min_valid_ratio >= valid_ratio_threshold, (
		f"Minimum valid ratio {min_valid_ratio:.6f} is below threshold {valid_ratio_threshold:.6f}"
	)

	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run GREEDY reproduction metrics in PyTorch.")
	parser.add_argument(
		"--config_path",
		type=str,
		default="experiments/reproduction_experiments/greedy/reproduce_greedy.yml",
		help="Path to GREEDY reproduction config.",
	)
	args = parser.parse_args()

	run_reproduction(args.config_path)


