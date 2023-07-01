import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .exception import ModelWrapperError
from .model_wrapper import model_wrapper


class MLP(model_wrapper):
    def train(self, X, y):
        if self.__model is not None:
            self._init_model()

        self.__model.fit(X, y)
        self.__is_learned = True

    def predict_proba(self, X):
        if not self.__is_learned:
            raise ModelWrapperError("the model is not learned")

        return self.__model.predict_proba(X)

    def setup(self, parameter: dict):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def _init_model(self):
        self.__model = MLP_Torch(2)
        self.__is_learned = False


class MLP_Torch(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 1)
        self.F = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        x = self.F(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


def _fit(model: nn.Module, X, y, epoch: int = 3000, cuda: bool = False):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    if cuda:
        criterion = criterion.cuda()

    for i in tqdm(range(epoch)):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _predict(model: nn.Module, X):
    with torch.no_grad():
        pred = model(X)
    return pred
