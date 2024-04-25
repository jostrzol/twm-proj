from typing import BinaryIO

import cv2
import numpy as np

from twm_proj.interface.image_reader import IImageReader


class ImageReader(IImageReader):
    def read(self, file: BinaryIO) -> np.ndarray:
        buffer = np.asarray(bytearray(file.read()), dtype="uint8")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image
