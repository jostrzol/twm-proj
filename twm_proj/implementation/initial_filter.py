import cv2
import numpy as np

from twm_proj.interface.initial_filter import IInitialFilter


class InitialFilter(IInitialFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gaussian_size = (3, 3)
        gaussian_gray = cv2.GaussianBlur(gray, gaussian_size, 0)
        closed = cv2.morphologyEx(gaussian_gray, cv2.MORPH_CLOSE, (10, 10))

        return closed
