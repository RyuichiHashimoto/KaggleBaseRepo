from models.model_wrapper import ModelParameter, model_wrapper
from models.logistic_regression import LogisticRegression
from models.MLP_torch import MLP_Torch
from models.random_forest import RandomForest


def new_model_instance(model_name: str, parameter: ModelParameter) -> model_wrapper:
    if model_name == LogisticRegression.__name__:
        return LogisticRegression(parameter)
    elif model_name == MLP_Torch.__name__:
        return MLP_Torch(parameter)
    elif model_name == RandomForest.__name__:
        return RandomForest(parameter)
    else:
        raise ValueError(f"we cannot found {model_name} as a model name")
