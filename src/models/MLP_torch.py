import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import polars as pl
from .exception import ModelWrapperError, ModelAlreadyLearnedError
from .model_wrapper import modelBase, ModelParameter
from typing_extensions import TypeAlias
from typing import Iterable, Union, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class MLP_Parameter(ModelParameter):
    inputdim: int
    epochs: int
    lr: float
    cuda: bool = False


ARG_Parameter: TypeAlias = Iterable[Union[MLP_Parameter, Dict[str, Any]]]


class MLP(modelBase):
    def __init__(self, parameter: ARG_Parameter):
        self.parameter = MLP_Parameter.from_dict(parameter)
        self._init_model()
        self.__is_trained = False

    def train(self, X: pl.DataFrame, y: pl.DataFrame):
        if self.__model is None:
            self._init_model()
        if self.__is_trained:
            raise ModelAlreadyLearnedError()

        X_Tensor = torch.from_numpy(X.to_numpy()).float()
        y_Tensor = torch.from_numpy(y.to_numpy()).float()

        _train(self.__model, X_Tensor, y_Tensor, self.parameter)

        self.__is_trained = True

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if not self.__is_trained:
            raise ModelWrapperError("the model is not learned")

        X_Tensor = torch.from_numpy(X.to_numpy()).float()

        return _predict(self.__model, X_Tensor, self.parameter).numpy()

    def setup(self, parameter):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def _init_model(self):
        self.__model = MLP_Torch(self.parameter.inputdim)
        self.__is_trained = False


class MLP_Torch(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Torch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)
        self.F = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        x = self.F(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


def _train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, parameter: MLP_Parameter) -> None:
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameter.lr)

    if parameter.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        X = X.cuda()
        y = y.cuda()

    for i in tqdm(range(parameter.epochs)):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _predict(model: nn.Module, X: torch.Tensor, parameter: MLP_Parameter) -> torch.Tensor:
    if parameter.cuda:
        X = X.cuda()

    with torch.no_grad():
        pred = model(X)

    if parameter.cuda:
        pred = pred.cpu()
    return pred
