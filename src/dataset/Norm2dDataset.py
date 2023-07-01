from dataclasses import dataclass
from .dataset import Dataset
from typing import Tuple, Iterable, Union, List
from typing_extensions import TypeAlias
import numpy as np
import polars as pl

Value: TypeAlias = Union[float, int, None]
Vector: TypeAlias = Tuple[Value, Value]
Matrix: TypeAlias = Tuple[Vector, Vector]


@dataclass(frozen=True)
class Norm2dParameter:
    myu: Vector
    sigma: Matrix
    samples: int


class Norm2dDataset(Dataset):
    def __init__(self, parameters: Iterable[Norm2dParameter]):
        super().__init__()
        self.parameters = parameters

        self.X_COLUMNS: Tuple[str, str] = ("x1", "x2")
        self.Y_COLUMN: str = "y"
        self.__data: pl.DataFrame = pl.DataFrame({"x1": [], "x2": []})

        self._create_dataset()

    def _create_dataset(self) -> None:
        df_list = []

        for idx, param in enumerate(self.parameters):
            data = np.random.multivariate_normal(param.myu, param.sigma, param.samples)

            df = pl.DataFrame({self.X_COLUMNS[0]: data[:, 0], self.X_COLUMNS[1]: data[:, 1]})
            df = df.with_columns(pl.lit(str(idx)).alias(self.Y_COLUMN))

            df_list.append(df)

        self.__data = pl.concat(df_list)

    @property
    def data(self) -> pl.DataFrame:
        return self.__data

    @property
    def X(self) -> pl.DataFrame:
        return self.__data.select(self.X_COLUMNS)

    @property
    def Y(self) -> pl.DataFrame:
        return self.__data.select(self.Y_COLUMN)

    def __setitem__(self, key: str, value: Union[pl.Series, np.ndarray]):
        if type(value) is pl.Series:
            self._add_row(value.alias(key))
        elif type(value) is np.ndarray:
            self._add_row(pl.Series(value).alias(key))
        else:
            raise ValueError(f"you must only polars.Series or np.ndarray, not {type(value)}")

    def __getitem__(self, key: Union[str, List[str]]) -> pl.DataFrame:
        return self.__data.select(key)

    def _add_row(self, row: pl.Series) -> None:
        self.__data = self.__data.with_columns(row)
