from DRE.DRE import Parameter as DRE_Parameter, DRE
from typing import Union
from models.model_wrapper import ModelParameter
from dataset.dataset import Dataset
from dataclasses import dataclass
from models import factory


@dataclass(frozen=True)
class MLBasedDRE_Parameter(DRE_Parameter):
    """
    Parameters
    ----------
    Parameter : _type_
        _description_
    """

    model_name: str
    model_param: Union[ModelParameter, dict]


class MLBasedDRE(DRE):
    MIN_CORERCTION: float = 0.00001

    def __init__(self, param: Union[dict, DRE_Parameter]):
        self.parameter = MLBasedDRE_Parameter.from_dict(param)

    def estimate(self, dataset: Dataset) -> float:
        # model learning and pridict
        self.model = factory.new_model_instance("LogisticRegression", self.parameter.model_param)
        self.model.train(dataset.X, dataset.Y)

        dataset["rprob"] = self.model.predict_proba(dataset.X)[:, 1].round(1)  # クラス1に含む確率
        dataset["inv_rprob"] = 1 - dataset["rprob"]  # クラス2に含む確率

        dataset["DRE"] = dataset["rprob"] / dataset["inv_rprob"].clip(upper_bound=2, lower_bound=self.MIN_CORERCTION)

        return dataset["DRE"].mean()
