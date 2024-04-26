from abc import ABCMeta
from dataclasses import dataclass

import numpy as np


@dataclass
class RectangleImage:
    size: tuple[int, int]
    image: np.ndarray


class IRectTransformer(metaclass=ABCMeta):
    def transform(
        self, image: np.ndarray, rect: np.ndarray
    ) -> RectangleImage: ...
