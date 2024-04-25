from twm_proj.interface.rect_classifier import IRectClassifier, RectangleType


class RectClassifier(IRectClassifier):
    def classify(self, rect_size: tuple[int, int]) -> RectangleType:
        raise NotImplementedError()
