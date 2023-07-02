from dataclasses import dataclass
from typing import Tuple, Union, List
from typing_extensions import TypeAlias
from abc import abstractmethod
import numpy as np
import polars as pl

Value: TypeAlias = Union[float, int, None]
Vector: TypeAlias = Tuple[Value, Value]
Matrix: TypeAlias = Tuple[Vector, Vector]


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
