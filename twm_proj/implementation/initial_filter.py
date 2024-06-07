import cv2
import numpy as np

from twm_proj.interface.initial_filter import IInitialFilter


class InitialFilter(IInitialFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturated = self.color_filter(hsv, (100, 135), saturate=0.5, brighten=1.2)
        saturated = self.color_filter(saturated, (51, 90), saturate=0.5, brighten=5.)
        bgr = cv2.cvtColor(saturated, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gaussian_size = (3, 3)
        gaussian_gray = cv2.GaussianBlur(gray, gaussian_size, 0)

        equalized = cv2.equalizeHist(gaussian_gray)

        return equalized

    @staticmethod
    def color_filter(
        hsv: np.ndarray,
        hue_range: tuple[int, int],
        saturate: float = 1,
        brighten: float = 1,
    ) -> np.ndarray:
        hue_min, hue_max = hue_range
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1].astype(np.float64)
        value = hsv[:, :, 2].astype(np.float64)

        matches = np.logical_and(hue >= hue_min, hue < hue_max)
        saturation[matches] = saturation[matches] * saturate
        saturation[saturation > 255] = 255
        value[matches] = value[matches] * brighten
        value[value > 220] = 220
        hsv[:, :, 1] = saturation.astype(np.uint8)
        hsv[:, :, 2] = value.astype(np.uint8)
        return hsv
