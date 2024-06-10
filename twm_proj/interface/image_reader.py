from abc import ABCMeta
from typing import BinaryIO

import numpy as np


class IImageReader(metaclass=ABCMeta):
    def read(self, file: BinaryIO) -> np.ndarray: ...
