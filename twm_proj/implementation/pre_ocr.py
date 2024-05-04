import cv2
import numpy as np

from twm_proj.interface.pre_ocr import IPreOcr


class PreOcr(IPreOcr):

    def cut(self, image: np.ndarray) -> np.ndarray:
        # TODO: cut based on plate shape type
        left = 25
        right = 5
        top = 10
        bottom = 20
        return image[top:-bottom, left:-right]

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        return img
