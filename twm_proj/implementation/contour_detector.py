from typing import Any, Generator

import cv2
import numpy as np
from shapely import Polygon

from twm_proj.interface.contour_detector import IContourDetector


class ContourDetector(IContourDetector):
    def detect(self, image: np.ndarray) -> Generator[np.ndarray, Any, None]:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.reshape(-1, 2)
            hull = cv2.convexHull(contour).reshape(-1, 2)
            if len(hull) < 3:
                continue
            poly_hull = Polygon(hull)
            if poly_hull.area <= 1000:
                continue

            yield contour
