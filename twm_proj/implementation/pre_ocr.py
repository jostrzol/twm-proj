import cv2
import numpy as np

from twm_proj.interface.pre_ocr import IPreOcr
from twm_proj.interface.rect_classifier import RectangleType

ASPECT_RATIO = 60 / 80


class PreOcr(IPreOcr):

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        image = self._convert_reds(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 81, 255, cv2.THRESH_BINARY)
        return img

    def expand(self, image: np.ndarray) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        new_height, new_width = image_height + 4, image_width + 4
        blank_image = 255 * np.ones((new_height, new_width), np.uint8)
        start_y = (new_height - image_height) // 2
        start_x = (new_width - image_width) // 2
        end_y = start_y + image_height
        end_x = start_x + image_width
        blank_image[start_y:end_y, start_x:end_x] = image
        return blank_image

    def filter_grayscale(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        plate_filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        plate_filtered = cv2.morphologyEx(plate_filtered, cv2.MORPH_CLOSE, kernel)
        return plate_filtered

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
        image_height, image_width = image.shape[:2]
        new_width, new_height = round(image_height * ASPECT_RATIO), image_height
        if new_width < image_width:
            new_width, new_height = image_width, round(image_height / ASPECT_RATIO)
        blank_image = 255 * np.ones((new_height, new_width), np.uint8)

        start_y = (new_height - image_height) // 2
        start_x = (new_width - image_width) // 2
        end_y = start_y + image_height
        end_x = start_x + image_width

        blank_image[start_y:end_y, start_x:end_x] = image
        return blank_image

    def get_letters(self, image: np.ndarray, plate_cls: RectangleType):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = image.shape
        image_area = image_height * image_width
        # sort contours based on their center
        # sort left-right
        contours = sorted(contours, key=lambda x: cv2.minEnclosingCircle(x)[0][0])
        # sort up-down (roughly - upper/lower part of the image)
        # note: sort is stable and the order will be maintained in the next step
        two_row_multiplier = 1
        if plate_cls == RectangleType.TWO_ROW_PLATE:
            contours = sorted(
                contours,
                key=lambda x: cv2.minEnclosingCircle(x)[0][1] > image_height / 2,
            )
            # letter height is not compared to full img_height
            two_row_multiplier = 0.6
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # skip contour of whole image
            if abs(image_area - area) < image_area * 0.01:
                continue

            cut = image[y : y + h, x : x + w]
            if (
                image_height * 0.55 * two_row_multiplier <= h
                and image_height * image_width * 0.03 <= h * w
            ):
                yield self._expand_width(cut)
