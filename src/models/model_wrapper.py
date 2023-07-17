from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


class ModelParameter(metaclass=ABCMeta):
    @classmethod
    def from_dict(cls, data):
        """
        辞書から当該クラスを返す。もし当該クラスのインスタンスが来たらそのまま返す。

        Parameters
        ----------
        data : Union[Dict[str, Any], LogisticRegression_Parameter]

        Returns
        -------
        Norm2dParameter

        Raises
        ------
        TypeError
            辞書型もしくは当該インスタンス以外が引数に指定された場合に例外を返す。
        """
        if isinstance(data, dict):
            return cls(**data)
        elif isinstance(data, cls):
            return data
        else:
            raise TypeError("Expected dict or ExampleClass instance")


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
