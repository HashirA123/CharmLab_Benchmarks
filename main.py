# generic example of a full end to end run of the repo
from data.data_object import DataObject
from evaluation.catalog.distances import Distance
from model.model_object import ModelObject
from method.catalog.ROAR.method import ROAR
import numpy as np
import pandas as pd

if __name__ == "__main__":

    data_object = DataObject(
        data_path="data/raw_csv/german.csv",
        config_path="data/config_files/data_config_german.yml")
    
    print("here is the processed data:")
    print(data_object.get_processed_data().head())

    model_module = ModelObject(
        config_path="model/model_config_mlp.yml",
        data_object=data_object
    )

    # get model accuracy
    train_accuracy = model_module.get_train_accuracy()
    print(f"Model training accuracy: {train_accuracy}")
    accuracy = model_module.get_test_accuracy()
    print(f"Model test accuracy: {accuracy}")

    # test to see if ROAR method runs without error
    method = ROAR(data_object, model_module)

    # get some factuals to generate counterfactuals for
    X_test, y_test = model_module.get_test_data()

    # get the first 5 rows of the processed test data as factuals
    # specifically, we can the ones predicted as the negative class (label 0) 
    predictions = model_module.predict(X_test)
    negative_class_indices = np.where(predictions == 0)[0]

    factuals = pd.DataFrame(X_test[negative_class_indices][:5], columns=data_object.get_feature_names(expanded=True))

    print("Here are the factuals we will generate counterfactuals for:")
    print(factuals)

    # now generate counterfactuals for these factuals using ROAR
    counterfactuals = method.get_counterfactuals(factuals)
    print("Here are the generated counterfactuals:")
    print(counterfactuals)

    # perform some benchmarking of the method using the evaluation module
    evaluation_module = Distance(data_object)

    evaluation_results = evaluation_module.get_evaluation(factuals, counterfactuals)
    print("Here are the evaluation results for the generated counterfactuals:")
    print(evaluation_results)