from dataset.Norm2dDataset import Norm2dParameter, Norm2dDataset

if __name__ == "__main__":
    one = Norm2dParameter((0, 0), ((1, 0), (0, 1)), 1000)
    two = Norm2dParameter((0, 0), ((1, 0), (0, 1)), 1000)

    dataset = Norm2dDataset((one, two))
    print(dataset.X_COLUMNS)
    df = dataset.X
