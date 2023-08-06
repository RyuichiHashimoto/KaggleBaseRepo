from DRE.DRE import Parameter as DRE_Parameter, DRE
from typing import Union
from models.modelBase import ModelParameter
from dataset.dataset import Dataset
from dataclasses import dataclass
from models.modelFactory import ModelFactory


@dataclass(frozen=True)
class MLBasedDRE_Parameter(DRE_Parameter):
    """
    Parameters
    ----------
    Parameter : _type_
        _description_
    """

    model_param: Union[ModelParameter, dict]


class MLBasedDRE(DRE):
    MIN_CORERCTION: float = 0.00001

    def __init__(self, param: Union[dict, DRE_Parameter]):
        self.parameter = MLBasedDRE_Parameter.from_dict(param)

    def estimate(self, dataset: Dataset) -> float:
        self.model = ModelFactory.create(self.parameter.model_param)
        self.model.train(dataset.X, dataset.Y)

        dataset["rprob"] = self.model.predict_proba(dataset.X).reshape(-1).round(2)  # クラス1に含む確率
        dataset["inv_rprob"] = 1 - dataset["rprob"]  # クラス2に含む確率

        dataset["DRE"] = dataset["rprob"] / dataset["inv_rprob"].clip(upper_bound=2, lower_bound=self.MIN_CORERCTION)
        y1_size = dataset.count_value("y", 1)
        y0_size = dataset.count_value("y", 0)

        dataset["DRE"] = dataset["DRE"] * y1_size / y0_size
        return dataset["DRE"].mean()
