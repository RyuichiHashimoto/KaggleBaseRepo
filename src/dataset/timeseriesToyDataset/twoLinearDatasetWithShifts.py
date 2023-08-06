from dataclasses import dataclass
from dataset.dataset import Dataset, Parameter
from typing import Tuple, Union, List, Collection
import numpy as np
import polars as pl


@dataclass(frozen=True)
class LinearWithShiftsParameter(Parameter):
    n_shifts: int
    slope: float
    intercept: float
    noize: float
    size: int


def _create_linearData(param: LinearWithShiftsParameter) -> pl.DataFrame:
    linearData = np.linspace(0, param.size, param.size) * param.slope + param.intercept  # 線形データ
    noize = np.random.normal(loc=0, scale=np.sqrt(param.noize), size=param.size)  # ノイズ

    return linearData + noize


class twoLinearDatasetWithShifts(Dataset):
    def __init__(self, parameters: Tuple[LinearWithShiftsParameter, LinearWithShiftsParameter]):
        super().__init__()
        assert len(parameters) == 2
        assert parameters[0].n_shifts == parameters[1].n_shifts

        self.parameters = [LinearWithShiftsParameter.from_dict(param) for param in parameters]

        self.X_COLUMNS: Tuple[str] = tuple(["x"] + [f"x{i+1}" for i in range(parameters[1].n_shifts)])
        self.Y_COLUMN: str = "y"

        self._create_dataset()

    def _create_dataset(self) -> None:
        df_list = []
        last_idx = 0

        for idx, param in enumerate(self.parameters):
            data = _create_linearData(param)
            idxs = [last_idx + 1 + i for i in range(0, param.size)]
            df = pl.DataFrame({self.X_COLUMNS[0]: data, "idx": idxs})
            df = df.with_columns(pl.lit(int(idx)).alias(self.Y_COLUMN).cast(int))

            for i in range(param.n_shifts):
                df = df.with_columns(df.get_column("x").shift(i + 1).alias(f"x{i+1}"))

            df_list.append(df)

            last_idx += param.size

        self._data = pl.concat(df_list)

    def drop_nulls(self, subset: Union[str, Collection[str]]) -> None:
        """
        指定した列({column_name})にnullがあるレコードを削除する

        Parameters
        ----------
        column_name : str
            nullがあるかチェックする列
        """
        self._data = self._data.drop_nulls(subset)
