import cv2

import numpy as np

from twm_proj.interface.rect_detector import IRectDetector


class RectDetector(IRectDetector):
    def detect(self, contour: np.ndarray) -> np.ndarray | None:
        rect = cv2.approxPolyDP(contour, epsilon=70.0, closed=True)
        if len(rect) != 4:
            return None
        return rect.reshape(-1, 2)
