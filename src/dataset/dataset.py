from typing import Tuple, Union, List, Collection
from typing_extensions import TypeAlias
from abc import abstractmethod, ABCMeta

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
            raise TypeError("Expected dict or Parameter instance")


class Dataset(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.X_COLUMNS: Tuple[str, str] = ("x1", "x2")
        self.Y_COLUMN: str = "y"
        self._data: pl.DataFrame = pl.DataFrame({"x1": [], "x2": []})

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def X(self) -> pl.DataFrame:
        return self._data.select(self.X_COLUMNS)

    @property
    def Y(self) -> pl.DataFrame:
        return self._data.select(self.Y_COLUMN)

    def __setitem__(self, key: str, value: Union[pl.Series, np.ndarray]):
        if type(value) is pl.Series:
            self._add_row(value.alias(key))
        elif type(value) is np.ndarray:
            self._add_row(pl.Series(value).alias(key))
        else:
            raise ValueError(f"you must only polars.Series or np.ndarray, not {type(value)}")

    def __getitem__(self, key: Union[str, List[str]]) -> Union[pl.DataFrame, pl.Series]:
        """
        Retrieve one or multiple columns from the DataFrame.

        Parameters
        ----------
        key : Union[str, List[str]]
            If a string is provided, the function will return a single column with the given name as a pandas Series.
            If a list of strings is provided, the function will return a DataFrame that includes all specified columns.

        Returns
        -------
        Union[pl.DataFrame, pl.Series]
            A DataFrame or Series containing the selected columns.

        Raises
        ------
        TypeError
            If the key is neither a string nor a list of strings, a TypeError will be raised.
        """
        if type(key) is str:
            return self._data.get_column(key)
        elif type(key) is List[str]:
            return self._data.select(key)
        else:
            raise TypeError()

    def _add_row(self, row: pl.Series) -> None:
        self._data = self._data.with_columns(row)

    def count_value(self, key: str, value: Value) -> int:
        """
        Counts the number of occurrences of a specific value in a specified column of the DataFrame.

        This method filters the DataFrame based on the condition where the values in the specified column
        are equal to the given value. Then it counts the number of rows in the resulting DataFrame, which
        is equal to the number of occurrences of the given value.

        Parameters
        ----------
        key : str
            The name of the column to consider for counting the occurrences of the value.

        value : Value
            The value whose occurrences are to be counted.

        Returns
        -------
        int
            The number of occurrences of the specified value in the specified column.

        Example
        --------
        Assuming `df` is an instance of a class with this method, and it has a column "A" with values 1, 2, 2, 3.

        To count the occurrences of value 2 in column "A":

            count = df.count_value("A", 2)

        The variable `count` will be 2, because value 2 appears twice in column "A".

        """
        return self._data.filter(pl.col(key) == value).shape[0]

    def drop_nulls(self, subset: Union[str, Collection[str]]) -> None:
        """
        Removes the rows in the DataFrame which have null values in the specified subset of columns.

        The subset of columns to check for null values can either be a single column (a string),
        or a collection of columns (a list, tuple, or any other iterable of strings).

        Note that the changes are performed in-place, i.e., the original DataFrame is modified
        and the method does not return a new DataFrame.

        Parameters
        ----------
        subset : Union[str, Collection[str]]
            A single column or a collection of columns to consider for checking null values.

        Example
        --------
        Assuming `df` is an instance of a class with this method, and it has columns "A", "B", and "C".

        To remove rows with null values in column "A":

            df.drop_nulls("A")

        To remove rows with null values in any of the columns "A" and "B":

            df.drop_nulls(["A", "B"])

        """
        self._data = self._data.drop_nulls(subset)
