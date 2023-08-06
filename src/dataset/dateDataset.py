from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Literal, Tuple, Union

import numpy as np
import polars as pl
from typing_extensions import TypeAlias

from .dataset import Dataset, Parameter
from util.util_datetime import DailyDateIterator, to_yyyymmdd


@dataclass(frozen=True)
class DateParameter(Parameter):
    sensor: Literal["G", "H"]
    first_interval: Tuple[date, date]
    second_interval: Tuple[date, date]
    ports: Iterable[str]
    every: str = "10s"
    ignore_irrelevant: bool = False  # portsで指定していないポートの情報を保持するかいなか


ARG_Parameter: TypeAlias = Union[DateParameter, Dict[str, Any]]


class DateDataset(Dataset):
    def __init__(self, parameter: ARG_Parameter):
        super().__init__()
        self.parameters: DateParameter = DateParameter.from_dict(parameter)

        self.X_COLUMNS: Tuple[str, str] = ("all_packets", "target_packets")
        self.Y_COLUMN: str = "y"

        self._create_dataset()

    def _create_dataset(self) -> None:
        df_list = []

        for date in DailyDateIterator(*self.parameters.first_interval):
            df = self._create_dataset(date, self.parameters)
            df = df.with_columns(pl.lit(int(0)).alias(self.Y_COLUMN).cast(int))
            # df = df.with_columns(pl.lit(to_yyyymmdd(date)).alias("date").cast(str))
            df_list.append(df)

        for date in DailyDateIterator(*self.parameters.second_interval):
            df = self._create_dataset(date, self.parameters)
            df = df.with_columns(pl.lit(int(1)).alias(self.Y_COLUMN).cast(int))
            # df = df.with_columns(pl.lit(to_yyyymmdd(date)).alias("date").cast(str))
            df_list.append(df)

        self._data = pl.concat(df_list)
        # self.__data = self.__data.with_columns(pl.lit(20).alias("all_packets").cast(int))
        # self.__data = self.__data.with_columns(pl.lit(10).alias("target_packets").cast(int))
