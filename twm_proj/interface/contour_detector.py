from abc import ABCMeta
from typing import Any, Generator

import numpy as np


class IContourDetector(metaclass=ABCMeta):
    # TODO: specify return type
    def detect(self, image: np.ndarray) -> Generator[Any, Any, None]: ...
