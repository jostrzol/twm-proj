from abc import ABCMeta

import numpy as np


class IOcr(metaclass=ABCMeta):
    def scan_text(self, image: np.ndarray) -> str: ...
