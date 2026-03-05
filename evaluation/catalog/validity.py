
import pandas as pd

from evaluation.evaluation_factory import register_evaluation
from evaluation.evaluation_object import EvaluationObject
from evaluation.utils import remove_nans


@register_evaluation("Validity")
class Validity(EvaluationObject):
    def __init__(self, data, model, hyperparameters = None):
        super().__init__(data, model, hyperparameters)


    def get_evaluation(self, factuals, counterfactuals):
        # keep valid counterfactuals and corresponding factuals
        counterfactuals_without_nans, _ = remove_nans(
            counterfactuals, factuals
        )

        if len(counterfactuals_without_nans) == 0:
            return 0.0  # No valid counterfactuals, so validity is 0
        
        # Validity is the percentage of counterfactuals that are predicted as the target class by the model
        target = 0.5

        # convert counterfactuals_without_nans to numpy array if it's a DataFrame
        if isinstance(counterfactuals_without_nans, pd.DataFrame):
            # the counterfactuals have the label column, we need to drop it before passing to the model
            counterfactuals_without_nans = counterfactuals_without_nans.drop(columns=[self.data.get_target_column()]).to_numpy()

        validity_score = sum(self.model.predict_proba(counterfactuals_without_nans)[:, 1] >= target) / len(counterfactuals)

        return validity_score

