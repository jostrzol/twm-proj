import cv2
import numpy as np
import pytesseract

from twm_proj.interface.ocr import IOcr


class Ocr(IOcr):
    WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%"

    def scan_text(self, image: np.ndarray) -> str:
        text: str = pytesseract.image_to_string(
            image.astype(np.uint8),
            config=f"--psm 7 -c tessedit_char_whitelist={self.WHITELIST}",
        )
        return text.strip()
