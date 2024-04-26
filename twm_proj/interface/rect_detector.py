from abc import ABCMeta

import numpy as np


class IRectDetector(metaclass=ABCMeta):
    def detect(self, contour: np.ndarray) -> np.ndarray | None: ...
