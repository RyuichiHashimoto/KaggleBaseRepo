from dataclasses import dataclass
from dataset.dataset import Dataset, Parameter
from typing import Tuple, Iterable, Union, List
import numpy as np
import polars as pl


@dataclass(frozen=True)
class timeSeriesParameter(Parameter):
    slope: float
    intercept: float
    noize: float
    size: int
    period: float = 1.0
    amplitude: float = 0.0


def _create_linearData(param: timeSeriesParameter) -> pl.DataFrame:
    LinearData = np.linspace(0, param.size, param.size) * param.slope + param.intercept  # 線形データ
    sin_data = param.amplitude * np.sin(np.linspace(0, param.size, param.size) * 2 * np.pi / param.period)
    noize = np.random.normal(loc=0, scale=np.sqrt(param.noize), size=param.size)  # ノイズ

    return sin_data + LinearData + noize


class timeSeriesDataset(Dataset):
    def __init__(self, parameters: Tuple[timeSeriesParameter, timeSeriesParameter]):
        super().__init__()
        assert len(parameters) == 2
        self.parameters = [timeSeriesParameter.from_dict(param) for param in parameters]

        self.X_COLUMNS: Tuple[str] = ("x",)
        self.Y_COLUMN: str = "y"
        self._data: pl.DataFrame = pl.DataFrame({"x": [], "idx": [], "y": []})

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

        self._data = pl.concat(df_list)
