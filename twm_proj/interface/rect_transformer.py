from abc import ABCMeta
from dataclasses import dataclass

import numpy as np

from twm_proj.interface.rect_detector import SlantedRectangle


@dataclass
class RectangleImage:
    size: tuple[int, int]
    image: np.ndarray


class IRectTransformer(metaclass=ABCMeta):
    def transform(
        self, image: np.ndarray, rect: SlantedRectangle
    ) -> RectangleImage: ...
