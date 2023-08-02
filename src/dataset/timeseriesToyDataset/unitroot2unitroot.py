from dataclasses import dataclass
from dataset.dataset import Dataset
from typing import Tuple, Iterable, Union, List
import numpy as np
import polars as pl
from typing_extensions import TypeAlias
from dataset.timeseriesToyDataset.parameters import UnitRootParameter
from typing import Dict, Any

ARG_Parameter: TypeAlias = Tuple[Union[UnitRootParameter, Dict[str, Any], Union[UnitRootParameter, Dict[str, Any]]]]


class UnitRoot2UnitRootDataset(Dataset):
    def __init__(self, parameters: Iterable[ARG_Parameter]):
        super().__init__()
        assert len(parameters) == 2
        self.parameters = [UnitRootParameter.from_dict(parameters[0]), UnitRootParameter.from_dict(parameters[1])]

        self.X_COLUMNS: Tuple[str] = ("x",)
        self.Y_COLUMN: str = "y"
        self.__data: pl.DataFrame = pl.DataFrame({"x": [], "idx": [], "y": []})

        self._create_dataset()

    def _create_dataset(self) -> None:
        df_list = []
        last_idx = 0

        param1: UnitRootParameter = self.parameters[0]
        noize = np.random.normal(loc=0, scale=np.sqrt(param1.var), size=param1.size)
        data = np.linspace(0, param1.size, param1.size) * param1.slope + param1.intercept + noize
        idxs = [last_idx + 1 + i for i in range(0, param1.size)]
        last_idx += param1.size
        df1 = pl.DataFrame({self.X_COLUMNS[0]: data, "idx": idxs})
        df1 = df1.with_columns(pl.lit(int(0)).alias(self.Y_COLUMN).cast(int))

        param2: UnitRootParameter = self.parameters[1]
        noize = np.random.normal(loc=0, scale=np.sqrt(param2.var), size=param2.size)
        data = np.linspace(0, param2.size, param2.size) * param2.slope + param2.intercept + noize
        idxs = [last_idx + 1 + i for i in range(0, param2.size)]
        last_idx += param2.size
        df2 = pl.DataFrame({self.X_COLUMNS[0]: data, "idx": idxs})
        df2 = df2.with_columns(pl.lit(int(0)).alias(self.Y_COLUMN).cast(int))

        self.__data = pl.concat([df1, df2])

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
