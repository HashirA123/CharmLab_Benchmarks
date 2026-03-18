from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge
from method.catalog.TREX.library.utils import TreXCounterfactualTorch
from method.method_factory import register_method
from method.method_object import MethodObject
from model.model_object import ModelObject


class _TreXLogitWrapper(nn.Module):
    """Normalize model outputs to logits expected by the TreX backend."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)

        if out.ndim == 1:
            out = out.unsqueeze(1)

        if out.ndim != 2:
            raise ValueError(f"Unexpected model output rank {out.ndim}. Expected rank 2.")

        # Convert binary scalar outputs to two-class logits [class_0, class_1].
        if out.shape[1] == 1:
            probs_pos = out
            if probs_pos.min() < 0.0 or probs_pos.max() > 1.0:
                probs_pos = torch.sigmoid(probs_pos)

            probs_pos = probs_pos.clamp(1e-6, 1.0 - 1e-6)
            probs = torch.cat([1.0 - probs_pos, probs_pos], dim=1)
            return torch.log(probs)

        if out.shape[1] >= 2:
            row_sums = out.sum(dim=1, keepdim=True)
            looks_like_probabilities = (
                torch.all(out >= 0.0)
                and torch.all(out <= 1.0)
                and torch.allclose(
                    row_sums,
                    torch.ones_like(row_sums),
                    atol=1e-3,
                    rtol=1e-3,
                )
            )

            if looks_like_probabilities:
                return torch.log(out.clamp(1e-6, 1.0))

            return out

        raise ValueError(f"Unsupported output shape: {tuple(out.shape)}")


@register_method("TREX")
class TreX(MethodObject):
    """
    Implementation of TreX [1]_.

    .. [1] Hamman, Faisal and Noorani, Erfaun and Mishra, Saumitra and Magazzeni, Daniele and Dutta, Sanghamitra
    Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees
    """

    def __init__(self, data: DataObject, 
                model: ModelObject,
                config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data, model, config_override=config_override)

        config = yaml.safe_load(open("method/catalog/TREX/library/config.yml", "r"))
        self.config = config if config is not None else {}
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        # store the feature ordering
        self._feature_order = self._data.get_feature_names(expanded=True)

        if not isinstance(self._model, nn.Module):
            raise ValueError("TREX currently supports only PyTorch nn.Module models.")

        self._apply_trex = self._coerce_bool(self.config.get("apply_trex", True))
        self._batch_size = int(self.config.get("batch_size", 1))

        clamp = self._resolve_clamp(self.config.get("clamp"))
        trex_p = self._parse_p_norm(self.config.get("trex_p", 2))
        norm = int(self._parse_p_norm(self.config.get("norm", 2)))
        num_classes = int(self.config.get("num_classes", self._infer_num_classes()))

        robust_class_default = 1 if num_classes > 1 else 0
        robust_class = int(self.config.get("robust_class", robust_class_default))
        robust_class = max(0, min(robust_class, num_classes - 1))

        wrapped_model = _TreXLogitWrapper(self._model)

        self._trex_generator = TreXCounterfactualTorch(
            model=wrapped_model,
            input_dim=len(self._feature_order),
            num_classes=num_classes,
            clamp=clamp,
            norm=norm,
            cf_steps=int(self.config.get("cf_steps", 60)),
            cf_step_size=float(self.config.get("cf_step_size", 0.02)),
            cf_confidence=float(self.config.get("cf_confidence", 0.5)),
            tau=float(self.config.get("tau", 0.75)),
            K=int(self.config.get("K", 1000)),
            sigma=float(self.config.get("sigma", 0.05)),
            trex_max_steps=int(self.config.get("trex_max_steps", 20)),
            trex_epsilon=float(self.config.get("trex_epsilon", 1.0)),
            trex_step_size=float(self.config.get("trex_step_size", 0.01)),
            trex_p=trex_p,
            batch_size=self._batch_size,
            robust_class=robust_class,
            model_outputs_logits=True,
            device=self.config.get("device"),
        )

    def _infer_num_classes(self) -> int:
        x_train, _ = self._model.get_train_data()
        if isinstance(x_train, pd.DataFrame):
            sample = x_train[self._feature_order].iloc[:1]
        else:
            sample = np.asarray(x_train, dtype=np.float32)[:1]

        proba = np.asarray(self._model.predict_proba(sample))
        if proba.ndim == 1:
            return 2
        if proba.shape[1] == 1:
            return 2
        return int(proba.shape[1])

    def _resolve_clamp(self, clamp_config: Any) -> Tuple[Any, Any]:
        if clamp_config is None:
            x_train, _ = self._model.get_train_data()

            if isinstance(x_train, pd.DataFrame):
                x_np = x_train[self._feature_order].to_numpy(dtype=np.float32)
            else:
                x_np = np.asarray(x_train, dtype=np.float32)

            return x_np.min(axis=0), x_np.max(axis=0)

        if isinstance(clamp_config, dict):
            if "low" not in clamp_config or "high" not in clamp_config:
                raise ValueError("TREX clamp config dict must include 'low' and 'high'.")
            return clamp_config["low"], clamp_config["high"]

        if isinstance(clamp_config, (list, tuple)) and len(clamp_config) == 2:
            return clamp_config[0], clamp_config[1]

        raise ValueError("TREX clamp must be None, a 2-item list/tuple, or a {'low','high'} dict.")

    @staticmethod
    def _parse_p_norm(value: Any):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"inf", "infinity", "np.inf"}:
                return np.inf
            if lowered in {"1", "2"}:
                return int(lowered)
        return value

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generate counterfactual examples for given factuals."""
        if factuals.shape[0] == 0:
            raise ValueError("Factuals should not be empty")

        factuals = factuals[self._feature_order]
        x_np = factuals.to_numpy(dtype=np.float32)

        original_pred = np.asarray(self._model.predict(factuals), dtype=np.int64).reshape(-1)

        x_cf, _, _ = self._trex_generator.generate(
            x_np=x_np,
            original_pred=original_pred,
            apply_trex=self._apply_trex,
        )

        df_cfs = pd.DataFrame(x_cf, columns=self._feature_order, index=factuals.index)
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index)
        return df_cfs
