from abc import ABCMeta, abstractmethod


class model_wrapper(metaclass=ABCMeta):
    """torchやsklaernなどの細かな実装違いを吸収するためのクラス"""

    __model = None
    __is_learned: bool = False

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def setup(self, parameter):
        pass

    @abstractmethod
    def close(self):
        pass

    def is_learned(self) -> bool:
        return self.__is_learned
