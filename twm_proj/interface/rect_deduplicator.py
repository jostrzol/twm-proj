from abc import ABCMeta

import numpy as np


class IRectDeduplicator(metaclass=ABCMeta):
    def reset(self): ...
    def is_dupe(self, rect: np.ndarray) -> bool: ...
