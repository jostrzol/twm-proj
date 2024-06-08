import cv2

import numpy as np

from twm_proj.interface.rect_detector import IRectDetector


class RectDetector(IRectDetector):
    @staticmethod
    def _sort_coordinates(coords: np.ndarray):
        cx, cy = coords.mean((0, 1))
        x, y = coords.T
        angles = np.arctan2(x - cx, y - cy)
        indices = np.argsort(angles)
        return coords[indices]

    def detect(self, contour: np.ndarray) -> np.ndarray | None:
        hull = cv2.convexHull(contour)
        rect = cv2.approxPolyDP(hull, epsilon=20.0, closed=True)
        if len(rect) != 4:
            return None
        return self._sort_coordinates(rect).reshape(-1, 2)
