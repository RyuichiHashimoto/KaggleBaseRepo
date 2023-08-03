from dataclasses import dataclass
from dataset.dataset import Dataset, Parameter
from typing import Tuple, Iterable, Union, List
import numpy as np
import polars as pl


@dataclass(frozen=True)
class LinearParameter(Parameter):
    slope: float
    intercept: float
    noize: float
    size: int


def _create_linearData(param: LinearParameter) -> pl.DataFrame:
    linearData = np.linspace(0, param.size, param.size) * param.slope + param.intercept  # 線形データ
    noize = np.random.normal(loc=0, scale=np.sqrt(param.noize), size=param.size)  # ノイズ

    return linearData + noize


class twoLinearDataset(Dataset):
    def __init__(self, parameters: Tuple[LinearParameter, LinearParameter]):
        super().__init__()
        assert len(parameters) == 2
        self.parameters = [LinearParameter.from_dict(param) for param in parameters]

        self.X_COLUMNS: Tuple[str] = ("x",)
        self.Y_COLUMN: str = "y"
        self.__data: pl.DataFrame = pl.DataFrame({"x": [], "idx": [], "y": []})

        self._create_dataset()

    def _create_dataset(self) -> None:
        df_list = []
        last_idx = 0

        for idx, param in enumerate(self.parameters):
            data = _create_linearData(param)
            idxs = [last_idx + 1 + i for i in range(0, param.size)]
            df = pl.DataFrame({self.X_COLUMNS[0]: data, "idx": idxs})
            df = df.with_columns(pl.lit(int(idx)).alias(self.Y_COLUMN).cast(int))
            df_list.append(df)
            last_idx += param.size

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
        if type(key) is str:
            return self.__data.get_column(key)
        elif type(key) is List[str]:
            return self.__data.select(key)
        else:
            raise TypeError()

    def _add_row(self, row: pl.Series) -> None:
        self.__data = self.__data.with_columns(row)
