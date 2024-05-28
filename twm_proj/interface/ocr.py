from abc import ABCMeta

import numpy as np


class IOcr(metaclass=ABCMeta):
    def scan_text(self, letters: list[np.ndarray]) -> str: ...
