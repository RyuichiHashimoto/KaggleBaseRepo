from dataset.timeseriesToyDataset.twoLinearDataset import LinearParameter
from dataset.timeseriesToyDataset.timeSeriesDataset import timeSeriesParameter
from dataset.timeseriesToyDataset.timeSeriesDatasetWithShifts import timeSeriesWithShiftsParameter
from typing import Iterable


def add_shifts_to_linear_param(param: timeSeriesParameter, n_shift: int) -> timeSeriesWithShiftsParameter:
    return timeSeriesWithShiftsParameter(
        n_shift, param.slope, param.intercept, param.noize, param.size, param.period, param.amplitude
    )


def add_shifts_to_linear_params(
    parameters: Iterable[LinearParameter], n_shift: int
) -> tuple[timeSeriesWithShiftsParameter]:
    return tuple([add_shifts_to_linear_param(param, n_shift) for param in parameters])


ToyDatasetProblem = [
    (LinearParameter(0, 5, 0.1, 10000), LinearParameter(0, 14, 0.1, 10000)),
    (LinearParameter(0.0005, 2, 0.1, 10000), LinearParameter(0.0005, 10, 0.1, 10000)),
    (LinearParameter(0.001, 2, 0.1, 10000), LinearParameter(0.001, 4, 0.1, 10000)),
    (LinearParameter(0, 3, 0.1, 10000), LinearParameter(0.001, 3, 0.1, 10000)),
    (LinearParameter(0.001, 2, 0.1, 10000), LinearParameter(0, 12, 0.1, 10000)),
    (LinearParameter(0.0001, 2, 0.1, 10000), LinearParameter(0.001, 3, 0.1, 10000)),
    (LinearParameter(0.001, 3, 0.1, 10000), LinearParameter(-0.001, 13, 0.1, 10000)),
    (LinearParameter(0, 8, 0.1, 10000), LinearParameter(0, 8, 0.1, 10000)),
    (LinearParameter(0.0005, 3, 0.1, 10000), LinearParameter(0.0005, 8, 0.1, 10000)),
]
