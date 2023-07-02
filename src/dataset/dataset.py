from dataclasses import dataclass
from typing import Tuple, Union, List
from typing_extensions import TypeAlias
from abc import abstractmethod
import numpy as np
import polars as pl

Value: TypeAlias = Union[float, int, None]
Vector: TypeAlias = Tuple[Value, Value]
Matrix: TypeAlias = Tuple[Vector, Vector]


class Parameter:
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


class Dataset:
    def __init__(self) -> None:
        self.X_COLUMNS: Tuple[str, str] = ("x1", "x2")
        self.Y_COLUMN: str = "y"
        self.__data: pl.DataFrame = pl.DataFrame({"x1": [], "x2": []})

    @property
    @abstractmethod
    def data(self) -> pl.DataFrame:
        pass

    @property
    @abstractmethod
    def X(self) -> pl.DataFrame:
        pass

    @property
    @abstractmethod
    def Y(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: Union[pl.Series, np.ndarray]):
        pass

    @abstractmethod
    def __getitem__(self, key: Union[str, List[str]]) -> pl.DataFrame:
        return self.__data.select(key)

    @abstractmethod
    def _add_row(self, row: pl.Series) -> None:
        self.__data = self.__data.with_columns(row)
