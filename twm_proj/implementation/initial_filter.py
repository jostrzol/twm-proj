import numpy as np

from twm_proj.interface.initial_filter import IInitialFilter


class InitialFilter(IInitialFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
