import cv2
import numpy as np

from twm_proj.interface.initial_filter import IInitialFilter


class InitialFilter(IInitialFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        image = self._dim_saturated(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gaussian_size = (3, 3)
        gaussian_gray = cv2.GaussianBlur(gray, gaussian_size, 0)

        equalized = cv2.equalizeHist(gaussian_gray)

        return equalized

    def _dim_saturated(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = self._saturation_filter(hsv, (70, 255), 0.9)
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _saturation_filter(
        hsv: np.ndarray,
        saturation_range: tuple[int, int],
        brighten: float = 1,
    ) -> np.ndarray:
        sat_min, sat_max = saturation_range
        sat = hsv[:, :, 1].astype(np.float64)
        value = hsv[:, :, 2].astype(np.float64)

        matches = np.logical_and(sat >= sat_min, sat < sat_max)
        value[matches] = value[matches] * brighten
        value[sat > 255] = 255
        hsv[:, :, 2] = value.astype(np.uint8)
        return hsv
