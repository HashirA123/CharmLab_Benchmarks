from typing import Any, Dict, Optional
import copy

import pandas as pd
import dice_ml
import yaml
from data.data_object import DataObject
from experiment_utils import deep_merge
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject

from evaluation.utils import check_counterfactuals



@register_method("DICE")
class Dice(MethodObject):
	"""
	Implementation of Dice from Mothilal et.al. [1]_.

	NOTE: This method is implemented using the authors own publicly avalable code imported as a library.

	.. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
	"""

	def __init__(
		self,
		data: DataObject,
		model: ModelObject,
		config_override: Optional[Dict[str, Any]] = None,
	):
		super().__init__(data, model, config_override=config_override)
		backend = (
			self._model._config.get("backend", "").lower()
			if hasattr(self._model, "_config")
			else ""
		)
		if backend and backend != "pytorch":
			raise ValueError(
				f"DICE currently supports only PyTorch backend, got: {backend}"
			)

		self.config = yaml.safe_load(open("method/catalog/DICE/library/config.yml", "r"))
		
		if self._config_override is not None:
			self.config = deep_merge(self.config, self._config_override)

		self._feature_order = self._data.get_feature_names(expanded=True)

		self._continuous = self._data.get_continuous_features()
		cat = self._data.get_categorical_features(expanded=True)
		self._categorical = []
		for groups in cat:
			self._categorical.extend(groups)
		self._target = self._data.get_target_column()

		# 1. Grab the dataframe
		df = self._data.get_processed_data().copy()
		
		df = df.astype('float64')

		self.dice_data = dice_ml.Data(
			dataframe=df,
			continuous_features=self._continuous,
			outcome_name=self._target,
		)

		# Use a CPU copy for dice_ml so we don't mutate the shared model device.
		self._dice_torch_model = copy.deepcopy(self._model).to('cpu')
		self._dice_model = dice_ml.Model(model=self._dice_torch_model, backend="PYT")

		self._dice = dice_ml.Dice(self.dice_data, self._dice_model, method="random")

		self._num = int(self.config["num"])
		self._desired_class = int(self.config["desired_class"])
		self._posthoc_sparsity_param = float(self.config["posthoc_sparsity_param"])

	def get_counterfactuals(self, factuals: pd.DataFrame):
		"""Generate counterfactuals for input factual instances."""
		querry_instances = factuals.copy()
		querry_instances = querry_instances[self._feature_order]

		# NOTE: Keep query instances as object dtype to avoid a pandas>=2 assignment
		# failure inside dice_ml's random explainer when categorical samples are
		# string-typed (e.g., "0.0") and candidate columns are float-typed.
		querry_instances = querry_instances.astype(object)
		
		if not querry_instances.shape[0] > 0:
			raise ValueError("Factuals should not be empty")
		
		dice_exp = self._dice.generate_counterfactuals(
			querry_instances,
			total_CFs=self._num,
			desired_class=self._desired_class,
			posthoc_sparsity_param=self._posthoc_sparsity_param,
		)

		list_cfs = dice_exp.cf_examples_list

		df_cfs = pd.concat([cf.final_cfs_df for cf in list_cfs], ignore_index=True)
		df_cfs = df_cfs.apply(pd.to_numeric, errors="coerce")

		df_cfs = check_counterfactuals(self._model, self._data, df_cfs, querry_instances.index)
		df_cfs = df_cfs[self._feature_order]
		return df_cfs