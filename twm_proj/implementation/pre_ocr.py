from collections import Counter
import cv2
import numpy as np

from twm_proj.interface.pre_ocr import IPreOcr


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

    # def _is_same_color(self, region, threshold=0.7):
    #     pixels = region.reshape(-1, region.shape[-1])
    #     color_counts = Counter(map(tuple, pixels))
    #     _, count = color_counts.most_common(1)[0]
    #     percentage = count / pixels.shape[0]
    #     print(percentage)
    #     return percentage >= threshold
    
    def get_letters(self, image: np.ndarray):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        image_height, image_width = image.shape
        image_area = image_height * image_width
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # skip contour of whole image
            if abs(image_area - area) < image_area * 0.01:
                continue

            cut = image[y:y+h, x:x+w]
            if image_height * 0.35 <= h:
                yield cut
                # cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)

        # result = cv2.bitwise_or(image, cv2.bitwise_not(mask))
        # return result
