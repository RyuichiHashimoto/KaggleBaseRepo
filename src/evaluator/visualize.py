import seaborn as sns
from typing import Optional
import os
from dataset.dataset import Dataset


def joinplot(
    dataset: Dataset,
    x1: Optional[str] = None,
    x2: Optional[str] = None,
    hue: Optional[str] = None,
    output_file_path: Optional[str] = None,
) -> None:
    """joinplot（散布図＋ヒストグラム）を描画するための関数

    Parameters
    ----------
    dataset : Dataset
        データセットクラス

    x1 : str, optional
        joinplotするときのx軸
        もし設定されていない場合、Dataset.X_COLUMN[0]が代入される。

    x2 : str, optional
        joinplotするときのy軸
        もし設定されていない場合、Dataset.X_COLUMN[0]が代入される。

    hue : str, optional
        データの色分けするときに使用するラベル。
        もし設定されていない場合は、

    output_file_path: str, optional
        ファイル出力するときのパス
        もし指定していなければ、ファイル保存しない。
    """
    if x1 is None:
        x1 = dataset.X_COLUMNS[0]
    if x2 is None:
        x2 = dataset.X_COLUMNS[1]

    result = sns.jointplot(data=dataset.data, x=x1, y=x2, hue=hue)

    if output_file_path is not None:
        if len(output_file_path.split("/")) != 1:
            directory = "/".join(output_file_path.split("/")[:-1])
            os.makedirs(directory, exist_ok=True)

        result.savefig(output_file_path)


def histplot(
    dataset: Dataset, x: str, hue: Optional[str], bins: int = 30, output_file_path: Optional[str] = None
) -> None:
    """joinplot（散布図＋ヒストグラム）を描画するための関数

    Parameters
    ----------
    dataset : Dataset
        データセットクラス

    x : str, optional
        histplotするときのx軸

    hue : str, optional
        データの色分けするときに使用するラベル。
        もし設定されていない場合は、

    bins:
        データ範囲を等間隔に区切ったビンの数


    output_file_path: str, optional
        ファイル出力するときのパス
        もし指定していなければ、ファイル保存しない。
    """

    result = sns.histplot(dataset.data, x=x, hue=hue, bins=bins)

    if output_file_path is not None:
        if len(output_file_path.split("/")) != 1:
            directory = "/".join(output_file_path.split("/")[:-1])
            os.makedirs(directory, exist_ok=True)

        result.savefig(output_file_path)
