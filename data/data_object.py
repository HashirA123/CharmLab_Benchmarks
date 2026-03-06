import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import yaml
from typing import Dict, Optional, Tuple, List, Any

class DataObject:
    """
    A unified data ingestion and preprocessing pipeline for algorithmic recourse tasks.
    
    This module reads raw CSV data and a YAML configuration file to dynamically 
    construct a processed dataset. It handles feature encoding, scaling, class 
    balancing, and data splitting based on the constraints specified in the config.


    NOTE: this module will essentially take the place of the existing data module and dataset classes,
    and all the functionality in the loadData method will be transferred here as member functions.
    The "get_preprocessing()" acts like a controller that, based on configs, will call appropriate util 
    funtions. (think large if-else block).

    * Update (05/03/2026):
        - This class can now be extended for specific datasets and the get_preprocessing() 
        method can be overridden to implement dataset-specific preprocessing, making use of 
        any of the defined member functions of this parent class.

    Attributes:
        raw_df (pd.DataFrame): The original, unprocessed data loaded from CSV.
        config (Dict[str, Any]): The parsed YAML configuration dictionary.
        processed_df (pd.DataFrame): The data after all preprocessing steps are applied.
        target (str): The name of the target column in the dataset.
        metadata (Dict[str, Any]): Generated bounds, constraints, and structural info for features.
    """

    def __init__(self, data_path: str, config_path: str = None, config_override: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataObject by loading the raw data and configuration.

        Args:
            data_path (str): The file path to the raw CSV dataset.
            config_path (str): The file path to the YAML configuration file.
            config_override (Optional[Dict[str, Any]]): Optional dictionary of config overrides.
        """
        self._metadata = {}

        if config_path is not None:
            with open(config_path, 'r') as file:
                self._config = yaml.safe_load(file)
        else:
            self._config = {}

        # If a pre-merged config is given, use it entirely (it already contains overrides)
        if config_override is not None:
            self._config = config_override

        self._data_path = data_path

        # self.get_preprocessing() # This will execute the entire preprocessing pipeline and populate processed_df and metadata.
        

    def get_preprocessing(self) -> None:
        """
        Executes the main preprocessing pipeline based on the YAML configuration.
        
        This method acts as the orchestrator. It iterates through the configured 
        features and applies the necessary helper functions in the correct sequence:
        1. Applies encoding (one-hot or thermometer) to categorical/ordinal columns.
        2. Applies mathematical scaling (normalization/standardization).
        3. Balances the dataset classes if specified in the dataset config.
        4. Calculates and stores lower/upper bounds for all processed features.
        
        Returns:
            None. Modifies `self.processed_df` and `self.metadata` in place.
        """
        self._read_raw_data()
        self._apply_scaling()
        self._apply_encoding()
        self._balance_dataset()
        self._enforce_feature_order() # Ensure the final DataFrame columns are in the exact order specified by the config.

    def get_processed_data(self) -> pd.DataFrame:
        """
        Returns the fully processed DataFrame ready for model training and counterfactual search.

        Returns:
            pd.DataFrame: The preprocessed dataset with all transformations applied.
        """
        return self._processed_df
    
    def get_target_column(self) -> str:
        """
        Returns the name of the target column as specified in the YAML configuration.

        Returns:
            str: The target column name.
        """
        return self._target
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns the metadata dictionary containing feature constraints, bounds, and structural info.

        Returns:
            Dict[str, Any]: The metadata for all features.
        """
        return self._metadata
    
    def get_categorical_features(self, expanded: bool = True) -> List[str]:
        """
        Retrieves a list of categorical features based on the configuration.

        Args:
            expanded (bool):
                - If True, returns the expanded feature names after encoding (e.g., ['WorkClass_cat_0', 'WorkClass_cat_1']).
                - If False, returns the original base feature names before encoding (e.g., ['WorkClass']).
        Returns:
            List[str]: A list of feature names that are categorized as 'categorical' in the config.
        """
        categorical_features = []
        for feature in self._config['features']:
            if self._config['features'][feature]['type'] == 'categorical':
                if expanded:
                    # Add all the expanded feature names that start with the base feature name
                    categorical_features.append(self._config['features'][feature]['encoded_feature_names'])
                else:
                    categorical_features.append(feature)
        return categorical_features

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

        self._raw_df = pd.read_csv(self._data_path)
        self._processed_df = self._raw_df.copy() # This will be transformed in place through the preprocessing pipeline.

        # drop columns not defined in the config
        columns_to_drop = [col for col in self._raw_df.columns if col not in self._config['features'].keys()]
        self._processed_df = self._processed_df.drop(columns=columns_to_drop, errors='ignore')
        self._target = self._config['target_column']

        for feature in self._config['features']:
            if feature not in self._raw_df.columns:
                raise ValueError(f"Feature '{feature}' defined in config is not present in the raw dataset.")
            else:
                # right now the conifgs are just being stored as another dictionary
                # which isnt much different fomrjust storing as a whole config.
                # this may need to be updated in the future if necessary.
                self._metadata[feature] = self._config['features'][feature]

    def _apply_encoding(self) -> None:
        """
        Helper method to encode categorical and ordinal features.
        
        Reads the 'encode' key from the feature configuration.
        - If 'one-hot': Applies standard one-hot encoding.
        - If 'thermometer': Applies ordinal thermometer encoding (preserving order).
        Updates the dataframe columns to reflect the new sub-categorical/sub-ordinal features.
        """
        # loop through the features in the config and apply the appropriate encoding based on the "encode" key.
        for feature in self._config['features']:
            if self._config['features'][feature]['encode'] == 'one-hot':
                self._apply_one_hot_encoding(feature)
            elif self._config['features'][feature]['encode'] == 'thermometer':
                self._apply_thermometer_encoding(feature)

    def _apply_one_hot_encoding(self, feature_name: str) -> None:
        """
        Applies one-hot encoding to a specified categorical feature.

        Args:
            feature_name (str): The name of the feature to encode.
        """
        one_hot = pd.get_dummies(self._processed_df[feature_name], prefix=feature_name + "_cat", dtype=float)
        self._processed_df = pd.concat([self._processed_df.drop(columns=[feature_name]), one_hot], axis=1)

    def _apply_thermometer_encoding(self, feature_name: str) -> None:
        """
        Applies thermometer encoding to a specified ordinal feature.

        Args:
            feature_name (str): The name of the feature to encode.
        """
        # NOTE: needs to be implemented
        raise NotImplementedError("Thermometer encoding strategy is not yet implemented.")

    def _apply_scaling(self) -> None:
        """
        Helper method to apply numerical scaling to the dataset.
        
        Reads the 'preprocessing_strategy' from the dataset configuration.
        - If 'normalize': Applies min-max scaling to bound continuous features strictly to [0, 1].
        - If 'standardize': Scales features to mean 0 and standard deviation 1.
        (Note: Care should be taken with 'standardize' as it alters ranges in factual/counterfactual domains).
        """
        if self._config['preprocessing_strategy'] == 'normalize':
            # NOTE: needs to be implemented
            raise NotImplementedError("Normalization strategy is not yet implemented.")
        elif self._config['preprocessing_strategy'] == 'standardize':
            scaler = StandardScaler()

            for feature in self._config['features']:
                if self._config['features'][feature]['type'] == 'numerical' and self._config['features'][feature]['node_type'] == 'input':
                    self._processed_df[feature] = scaler.fit_transform(self._processed_df[[feature]])

    def _balance_dataset(self) -> None:
        """
        Helper method to balance the representation of target classes.
        
        If 'balance_classes' is true in the config, this method identifies the minority 
        class in the target column and subsamples the majority class to match its count 
        (or rounds down to a configured interval).
        """
        if self._config['balance_classes']:
            # NOTE: needs to be implemented
            raise NotImplementedError("Class balancing strategy is not yet implemented.")

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the fully preprocessed dataset into training and testing sets.
        
        Uses the 'train_split' ratio defined in the YAML configuration. It strictly 
        separates 'input' nodes (X) from 'output' nodes (y) and ignores 'meta' nodes 
        for training purposes.

        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features.
                - X_test (pd.DataFrame): Testing features.
                - y_train (pd.Series): Training targets.
                - y_test (pd.Series): Testing targets.
        """
        X = self._processed_df.drop(columns=self._config['target_column']) # [self.get_feature_names(expanded=True)]
        y = self._processed_df[self._config['target_column']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self._config['train_split'], random_state=42)
        return X_train, X_test, y_train, y_test

    def _enforce_feature_order(self) -> None:
        """
        Reorders the columns of the processed DataFrame to ensure strict consistency.
        
        This method is called at the very end of get_preprocessing(). It uses the 
        base 'feature_order' from the config and expands it based on the encoding 
        results. For example, if 'WorkClass' is second in the base order, all of its 
        dummy variables (WorkClass_cat_0, WorkClass_cat_1) will be grouped together 
        in their sorted, expanded order in the final DataFrame.
        """
        feature_order = self._config['feature_order']
        expanded_feature_names = self.get_feature_names(expanded=True)

        reordered_columns = []
        for feature in feature_order:
            expanded_features = sorted([f for f in expanded_feature_names if f.startswith(feature + "_")])
            
            if expanded_features:
                reordered_columns.extend(expanded_features)
            elif feature in expanded_feature_names:
                reordered_columns.append(feature)

        target = self._config['target_column']
        if target in self._processed_df.columns:
            reordered_columns.append(target)

        self._processed_df = self._processed_df.reindex(columns=reordered_columns)

    def get_feature_names(self, expanded: bool = True) -> List[str]:
        """
        Retrieves the exact, ordered list of feature names used in the dataset.
        
        Args:
            expanded (bool): 
                - If True, returns the feature names AFTER encoding 
                  (e.g., ['Age', 'WorkClass_cat_0', 'WorkClass_cat_1', ...]).
                  This is the exact order fed into the ML model.
                - If False, returns the raw base feature names BEFORE encoding
                  (e.g., ['Age', 'WorkClass', 'EducationLevel']).

        Returns:
            List[str]: The strictly ordered feature names.
        """
        if expanded:
            return [col for col in self._processed_df.columns if col != self._config['target_column']]
        else:
            return [f for f in self._config['features'].keys() if self._config['features'][f]['node_type'] == 'input']
        
    def get_feature_indices(self, feature_name: str) -> List[int]:
        """
        Maps a base feature name to its column indices in the expanded DataFrame.
        
        In recourse algorithms, if you want to perturb 'WorkClass', the optimizer 
        needs to know exactly which column indices in the tensor belong to 'WorkClass'.
        This method returns those indices based on the enforced feature order.
        
        Args:
            feature_name (str): The base name of the feature (e.g., 'WorkClass').
            
        Returns:
            List[int]: The corresponding index or indices in the expanded data array 
                       (e.g., [1, 2, 3]).
        """
        raise NotImplementedError("Feature index mapping is not yet implemented.")

    def _filter_and_impute(self) -> None:
        """
        Filters the raw dataframe to only include features defined in the YAML config.
        Applies specified imputation strategies (e.g., 'median', 'mean', 'mode', or 
        a constant like -1) to handle missing values before encoding begins.
        """
        pass

    def inverse_transform(self, x_processed: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the preprocessing pipeline on a given dataframe of samples.
        
        Crucial for algorithmic recourse: Takes counterfactuals generated in the 
        scaled/encoded space and maps them back to the original, human-readable 
        feature space. Reverses normalization/standardization and collapses one-hot 
        or thermometer encodings back into their original categorical/ordinal labels.

        Args:
            x_processed (pd.DataFrame): Data in the transformed space.

        Returns:
            pd.DataFrame: Data in the original feature space.
        """
        pass

