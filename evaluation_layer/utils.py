from typing import Tuple, Union

import pandas as pd
import numpy as np
from data_layer.data_object import DataObject
from model_layer.model_object import ModelObject
import logging


def check_counterfactuals(model: ModelObject, 
                          data: DataObject,
                          counterfactuals: pd.DataFrame,
                          factual_indices: pd.Index) -> pd.DataFrame:
    """
    Check if the generated counterfactuals are valid by ensuring 
    that they are classified as the target class by the model. If any 
    counterfactual is not classified as the target class, we can either 
    raise an error or remove it from the final output.

    Parameters
    ----------
    model: ModelObject
        The model module containing the trained model and its configuration.
    counterfactuals: pd.DataFrame
        The generated counterfactuals to be checked.
    factual_indices: pd.Index
        The indices of the original factual instances corresponding to the counterfactuals.
    Returns
    ------- 
    pd.DataFrame
        The valid counterfactuals that are classified as the target class by the model.
    """
    print(f"These are the predicted values for the counterfactual instances before checking {model.predict_proba(counterfactuals)}")
    counterfactuals[data._config["target_column"]] = np.argmax(model.predict_proba(counterfactuals), axis=1)
    # Change all wrong counterfactuals to nan
    counterfactuals.loc[counterfactuals[data._config["target_column"]] != 1, :] = np.nan

    return counterfactuals

def remove_nans(
    counterfactuals: pd.DataFrame, factuals: pd.DataFrame = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Remove instances for which a counterfactual could not be found.

    Parameters
    ----------
    counterfactuals:
        Has to be the same shape as factuals.
    factuals:
        Has to be the same shape as counterfactuals. (optional)

    Returns
    -------

    """
    # get indices of unsuccessful counterfactuals
    nan_idx = counterfactuals.index[counterfactuals.isnull().any(axis=1)]
    output_counterfactuals = counterfactuals.copy()
    output_counterfactuals = output_counterfactuals.drop(index=nan_idx)

    if factuals is not None:
        if factuals.shape[0] != counterfactuals.shape[0]:
            raise ValueError(
                "Counterfactuals and factuals should contain the same amount of samples"
            )
        output_factuals = factuals.copy()
        output_factuals = output_factuals.drop(index=nan_idx)
        return output_counterfactuals, output_factuals

    return output_counterfactuals