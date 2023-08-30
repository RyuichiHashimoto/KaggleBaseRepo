from abc import ABCMeta, abstractmethod
from dataset.dataset import Dataset


class Parameter(metaclass=ABCMeta):
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
            raise TypeError("Expected dict or Parameter instance")


class DRE(metaclass=ABCMeta):
    @abstractmethod
    def estimate(self, dataset: Dataset):
        pass
