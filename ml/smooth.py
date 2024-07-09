import os

import torch
import numpy as np
import numpy.typing as npt
from PIL import Image

from ml.model import CNN


def image2tensor(image: npt.NDArray[np.uint8]):
    return torch.from_numpy(image / 255.0).float().unsqueeze(0)


def main(image: npt.NDArray[np.uint8], filename: str):
    network = CNN(30)
    network.load_state_dict(torch.load("ml/state_dict.pt"))
    large_image = network.clean_output(image2tensor(image))

    root, _ = os.path.splitext(filename)
    large_image.save(f"{root}_30x.png")
