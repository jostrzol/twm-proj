from abc import ABCMeta

import numpy as np

from twm_proj.interface.rect_classifier import RectangleType


class IPreOcr(metaclass=ABCMeta):
    def to_grayscale(self, image: np.ndarray) -> np.ndarray: ...
    def filter_grayscale(self, image: np.ndarray) -> np.ndarray: ...
    def get_letters(self, image: np.ndarray, plate_cls: RectangleType): ...
