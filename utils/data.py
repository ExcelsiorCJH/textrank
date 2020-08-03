import json
import numpy as np

from .types_ import *

np.random.seed(42)


def get_data(data_path: str = "data/train.json", data_type: str = "cnndm") -> List[str]:
    """
    Arguments
    ---------
    data_path: str
        data path to load
    data_type: str
        options = ['cnn/dm', etc]

    Returns
    -------
    data: list of str
    """
    if data_type == "cnndm":
        rnd_idx = np.random.randint(1000)
        with open(data_path, "r", encoding="utf8") as f:
            data = [json.loads(line) for line in f]

        data = data[rnd_idx]  # get random cnn/dm sample
        data = data["doc"]  # get document
        data = data.split("\n")  # get sentences
    else:
        with open(data_path, "r") as f:
            data = f.read().split("\n")

    return data
