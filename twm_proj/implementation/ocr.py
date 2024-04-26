import cv2
import numpy as np
import pytesseract  # also required: sudo apt-get install tesseract-ocr
from PIL import Image

from twm_proj.interface.ocr import IOcr


class Ocr(IOcr):
    def scan_text(self, image: np.ndarray) -> str:
        # Grayscale image
        img = image.convert("L")
        _, img = cv2.threshold(np.array(img), 125, 255, cv2.THRESH_BINARY)
        img = Image.fromarray(img.astype(np.uint8))
        return pytesseract.image_to_string(
            image,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%",
        )
