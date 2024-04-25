from abc import ABCMeta

import numpy as np


class IInitialFilter(metaclass=ABCMeta):
    def filter(self, image: np.ndarray) -> np.ndarray: ...
