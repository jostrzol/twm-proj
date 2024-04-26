import numpy as np

from twm_proj.interface.rect_detector import IRectDetector, SlantedRectangle


class RectDetector(IRectDetector):
    def detect(self, contour: np.ndarray) -> SlantedRectangle | None:
        raise NotImplementedError()
