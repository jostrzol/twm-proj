from abc import ABCMeta

import numpy as np


class IEdgeFilter(metaclass=ABCMeta):
    def filter(self, image: np.ndarray) -> np.ndarray: ...
