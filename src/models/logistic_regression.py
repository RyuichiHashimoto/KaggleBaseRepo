from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression

from .exception import ModelWrapperError
from .model_wrapper import model_wrapper


class LogisticRegression(model_wrapper):
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
        self.__model = sklearn_LogisticRegression(penalty="l2")
        self.__is_learned = False
