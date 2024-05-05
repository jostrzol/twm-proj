import cv2
import numpy as np

from twm_proj.interface.pre_ocr import IPreOcr


class PreOcr(IPreOcr):

    def cut(self, image: np.ndarray) -> np.ndarray:
        # TODO: cut based on plate shape type
        left = 25
        right = 5
        top = 10
        bottom = 8
        return image[top:-bottom, left:-right]

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        return img

    def filter_by_size(self, image: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if 1000 <= area <= 3000:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)
        result = cv2.bitwise_or(image, cv2.bitwise_not(mask))
        return result
