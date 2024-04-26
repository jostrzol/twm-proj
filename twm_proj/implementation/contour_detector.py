from typing import Any, Generator

import numpy as np
import cv2

from twm_proj.interface.contour_detector import IContourDetector


class ContourDetector(IContourDetector):
    def detect(self, image: np.ndarray) -> Generator[np.ndarray, Any, None]:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        yield from contours
