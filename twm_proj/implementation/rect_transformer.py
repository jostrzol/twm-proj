import cv2

import numpy as np
from numpy.linalg import norm

from twm_proj.interface.rect_transformer import IRectTransformer


class RectTransformer(IRectTransformer):
    def transform(self, image: np.ndarray, rect: np.ndarray) -> np.ndarray:
        src = [tl, bl, br, tr] = rect.reshape((4, 2))

        width_top = norm(tl - tr)
        width_bot = norm(bl - br)
        width = max(int(width_top), int(width_bot))

        height_left = norm(tl - bl)
        height_right = norm(br - tr)
        height = max(int(height_left), int(height_right))

        dst = np.float32(
            [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
        )
        transform = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        return cv2.warpPerspective(image, transform, (width, height))
