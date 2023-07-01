from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union
from typing_extensions import TypeAlias
import numpy as np

Value: TypeAlias = Union[float, int]
Vector: TypeAlias = Union(List[Value], np.array)
Matrix: TypeAlias = Union[List[Vector], np.ndarray]


@dataclass(frozen=True)
class datasetParameter:
    """dumy用のデータクラス"""


class dataset(metaclass=ABCMeta):
    """torchやsklaernなどの細かな実装違いを吸収するためのクラス"""

    __data = None

    @abstractmethod
    def load_dataset(self, parameter: List[datasetParameter]) -> Tuple[Matrix, Vector]:
        pass
