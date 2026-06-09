from __future__ import annotations

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SimpleTransform:
    def __init__(self, size: int, train: bool = False, normalize: bool = True) -> None:
        self.size = int(size)
        self.train = train
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor


def build_transform(input_size: int, train: bool = False) -> SimpleTransform:
    return SimpleTransform(size=input_size, train=train, normalize=True)
