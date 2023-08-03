from dataset.timeseriesToyDataset.twoLinearDataset import LinearParameter


ToyDatasetProblem = [
    (LinearParameter(0, 5, 0.1, 10000), LinearParameter(0, 14, 0.1, 10000)),
    (LinearParameter(0.0005, 2, 0.1, 10000), LinearParameter(0.0005, 10, 0.1, 10000)),
    (LinearParameter(0, 3, 0.1, 10000), LinearParameter(0.001, 3, 0.1, 10000)),
    (LinearParameter(0.001, 2, 0.1, 10000), LinearParameter(0, 12, 0.1, 10000)),
    (LinearParameter(0.0001, 2, 0.1, 10000), LinearParameter(0.001, 3, 0.1, 10000)),
    (LinearParameter(0.001, 3, 0.1, 10000), LinearParameter(-0.001, 13, 0.1, 10000)),
    (LinearParameter(0, 8, 0.1, 10000), LinearParameter(0, 8, 0.1, 10000)),
    (LinearParameter(0.0005, 3, 0.1, 10000), LinearParameter(0.0005, 8, 0.1, 10000)),
]
