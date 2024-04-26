from abc import ABCMeta

import numpy as np


class IRectTransformer(metaclass=ABCMeta):
    def transform(self, image: np.ndarray, rect: np.ndarray) -> np.ndarray: ...
