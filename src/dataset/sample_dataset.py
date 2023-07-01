from dataclasses import dataclass
from typing import Tuple
from typing_extensions import TypeAlias
import numpy as np

from .dataset import Value, dataset, datasetParameter

Vector = Tuple[Value, Value]
Matrix = Tuple[Vector, Vector]


@dataclass(frozen=True)
class Norm2dParameter(datasetParameter):
    sample: int
    myu: Vector
    sigma: Matrix


class Norm2dDataset(dataset):
    """torchやsklaernなどの細かな実装違いを吸収するためのクラス"""

    def load_dataset(self, parameter: Tuple[Norm2dParameter]) -> Tuple[Matrix, Vector]:
        for idx, param in enumerate(parameter):
            data = np.random.multivariate_normal(param.myu, param.sigma, param.sample)


if __name__ == "__main__":
    Norm2dParameter(1000, (0, 0), ((0, 1), (1, 0)))
