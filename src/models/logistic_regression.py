from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
from dataclasses import dataclass
from typing_extensions import TypeAlias
from typing import Iterable, Union, Dict, Any
from .exception import ModelWrapperError
from .modelBase import ModelBase, ModelParameter
from models.modelFactory import ModelFactory


@dataclass(frozen=True)
class LogisticRegression_Parameter(ModelParameter):
    penalty: str


ARG_Parameter: TypeAlias = Iterable[Union[LogisticRegression_Parameter, Dict[str, Any]]]


@ModelFactory.register(LogisticRegression_Parameter)
class LogisticRegression(ModelBase):
    def __init__(self, parameter: ARG_Parameter):
        self.parameter = LogisticRegression_Parameter.from_dict(parameter)
        self.__model = self._init_model()

    def train(self, X, y):
        if self.__model is None:
            self._init_model()

        self.__model.fit(X, y)
        self.__is_learned = True

    def predict_proba(self, X):
        if not self.__is_learned:
            raise ModelWrapperError("the model is not learned")

        return self.__model.predict_proba(X)

    def setup(self, parameter: ARG_Parameter):
        self.param = LogisticRegression_Parameter.from_dict(parameter)
        self.__model = self._init_model()

    def close(self):
        raise NotImplementedError()

    def _init_model(self):
        self.__model = sklearn_LogisticRegression(penalty=self.parameter.penalty)
        self.__is_learned = False
