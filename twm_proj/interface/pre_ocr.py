from abc import ABCMeta

import numpy as np


class IPreOcr(metaclass=ABCMeta):
    def cut(self, image: np.ndarray) -> np.ndarray: ...
    def to_grayscale(self, image: np.ndarray) -> np.ndarray: ...
