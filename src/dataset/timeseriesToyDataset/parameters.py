from dataclasses import dataclass

from dataset.dataset import Parameter


@dataclass(frozen=True)
class SteadyParameter(Parameter):
    mean: float
    var: float
    size: int


@dataclass(frozen=True)
class UnitRootParameter(Parameter):
    slope: float
    size: int
    var: float
    intercept: float = 0
