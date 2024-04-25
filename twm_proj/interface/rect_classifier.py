from abc import ABCMeta
from enum import Enum, auto


class RectangleType(Enum):
    ONE_ROW = auto()
    TWO_ROW = auto()
    NOT_PLATE = auto()


class IRectClassifier(metaclass=ABCMeta):
    def classify(self, rect_size: tuple[int, int]) -> RectangleType: ...
