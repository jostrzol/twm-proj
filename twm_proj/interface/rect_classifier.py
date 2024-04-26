from abc import ABCMeta
from enum import Enum, auto


class RectangleType(Enum):
    ONE_ROW_PLATE = auto()
    TWO_ROW_PLATE = auto()
    NOT_PLATE = auto()


class IRectClassifier(metaclass=ABCMeta):
    def classify(self, rect_size: tuple[int, int]) -> RectangleType: ...
