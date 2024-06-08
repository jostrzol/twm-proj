import cv2
import numpy as np
from shapely import Polygon

from twm_proj.interface.rect_detector import IRectDetector


class RectDetector(IRectDetector):
    @staticmethod
    def _sort_coordinates(coords: np.ndarray):
        cx, cy = coords.mean(0)
        x, y = coords.T
        angles = np.arctan2(x - cx, y - cy)
        indices = np.argsort(angles)
        return coords[indices]

    def detect(self, contour: np.ndarray) -> np.ndarray | None:
        hull = cv2.convexHull(contour).reshape(-1, 2)
        rect = self._approx_rect(hull)
        if rect is None:
            return None
        poly_contour = Polygon(contour)
        poly_rect = Polygon(rect)
        intersection = poly_contour.intersection(poly_rect)
        if intersection.area / poly_contour.area < 0.9:
            return None
        return self._sort_coordinates(rect)

    EPSILONS = [10.0, 20.0, 40.0]

    @classmethod
    def _approx_rect(cls, contour: np.ndarray) -> np.ndarray | None:
        for epsilon in cls.EPSILONS:
            approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
            approx = approx.reshape(-1, 2)
            if len(approx) == 4:
                return approx.reshape(-1, 2)
        return None
