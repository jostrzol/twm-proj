from collections import Counter
import cv2
import numpy as np

from twm_proj.interface.pre_ocr import IPreOcr

ASPECT_RATIO = 60 / 80


class PreOcr(IPreOcr):

    def cut(self, image: np.ndarray) -> np.ndarray:
        # TODO: cut based on plate shape type
        left = 20
        right = 5
        top = 10
        bottom = 8
        return image[top:-bottom, left:-right]

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        image = self._convert_reds(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        return img

    def _convert_reds(self, image: np.ndarray) -> np.ndarray:
        # Converts reds to blacks (special case for temporary plates)
        # Because of this, we can use stronger treshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 55, 50])
        upper_red = np.array([25, 255, 255])
        lower_red2 = np.array([160, 55, 100])
        upper_red2 = np.array([190, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        image[mask != 0] = [0, 0, 0]
        return image

    # train images are 80x60, so maintain that aspect ratio
    def _expand_width(self, image: np.ndarray):
        image_height, _ = image.shape
        new_width, new_height = round(image_height * ASPECT_RATIO), image_height
        blank_image = 255 * np.ones((new_height, new_width), np.uint8)

        original_height, original_width = image.shape[:2]
        start_y = (new_height - original_height) // 2
        start_x = (new_width - original_width) // 2

        blank_image[
            start_y : start_y + original_height, start_x : start_x + original_width
        ] = image
        return blank_image

    def get_letters(self, image: np.ndarray):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        image_height, image_width = image.shape
        image_area = image_height * image_width
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # skip contour of whole image
            if abs(image_area - area) < image_area * 0.01:
                continue

            cut = image[y : y + h, x : x + w]
            if image_height * 0.35 <= h and image_height * image_width * 0.03 <= h * w:
                yield self._expand_width(cut)
