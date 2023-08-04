from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
from dataclasses import dataclass
from typing_extensions import TypeAlias
from typing import Iterable, Union, Dict, Any
from .exception import ModelWrapperError
from .model_wrapper import modelBase, ModelParameter
from sklearn.ensemble import RandomForestClassifier as RF

try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except ImportError:
    pass


@dataclass(frozen=True)
class RandomForest_Parameter(ModelParameter):
    cuda: bool = False


ARG_Parameter: TypeAlias = Iterable[Union[RandomForest_Parameter, Dict[str, Any]]]


class RandomForest(modelBase):
    def __init__(self, parameter: ARG_Parameter):
        self.parameter = RandomForest_Parameter.from_dict(parameter)
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
        self.param = RandomForest_Parameter.from_dict(parameter)
        self.__model = self._init_model()

    def close(self):
        raise NotImplementedError()

    def _init_model(self):
        if self.parameter.cuda:
            self.__model = cuRF()
        else:
            self.__model = RF()
        self.__is_learned = False
