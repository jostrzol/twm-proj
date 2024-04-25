from abc import ABCMeta
from dataclasses import dataclass
from typing import Any


@dataclass
class SlantedRectangle:
    x1: tuple[int, int]
    x2: tuple[int, int]
    x3: tuple[int, int]
    x4: tuple[int, int]


class IRectDetector(metaclass=ABCMeta):
    # TODO: specify contour type
    def detect(self, contour: Any) -> SlantedRectangle | None: ...
