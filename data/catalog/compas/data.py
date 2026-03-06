import pandas as pd
from typing import Optional, Dict, Any
from data.data_object import DataObject


class CompasData(DataObject):

    def __init__(self, data_path: str, config_path: str = None, config_override: Optional[Dict[str, Any]] = None):
        super().__init__(data_path, config_path, config_override)
        
        self.get_preprocessing()

    def get_preprocessing(self):
        """
        Executes the main preprocessing pipeline based on the YAML configuration.
        """
        self._read_raw_data()
        self._apply_scaling()
        self._apply_encoding()
        self._balance_dataset()
        self._enforce_feature_order()

    def _read_raw_data(self):
        """
        Read the raw data from the CSV file and store relevant features and metadata.

        * This method must create the following member variables:
            - self._raw_df: The original, unprocessed data loaded from CSV.
            - self._processed_df: The data after all preprocessing steps are applied.
            - self._target: The name of the target column in the dataset.
            - self._metadata: Generated bounds, constraints, and structural info for features.

        How you go about creating these member variables is up to you, but they must be created by the end of this method.
        and must follow specifications of the parent docstring. You can create additional member variables
        if you need them.
        """

        self._raw_df = pd.read_csv(self._data_path).sample(frac=1, random_state=1).reset_index(drop=True)
        self._processed_df = self._raw_df.copy() # This will be transformed in place through the preprocessing pipeline.

        # drop columns not defined in the config
        columns_to_drop = [col for col in self._raw_df.columns if col not in self._config['features'].keys()]
        self._processed_df = self._processed_df.drop(columns=columns_to_drop, errors='ignore')
        self._target = self._config['target_column']

        for feature in self._config['features']:
            if feature not in self._raw_df.columns:
                raise ValueError(f"Feature '{feature}' defined in config is not present in the raw dataset.")
            else:
                self._metadata[feature] = self._config['features'][feature]
    
    

