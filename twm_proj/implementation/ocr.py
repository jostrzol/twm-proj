import cv2
import numpy as np
import pytesseract

from twm_proj.interface.ocr import IOcr


class Ocr(IOcr):
    WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%"

    def scan_text(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        text: str = pytesseract.image_to_string(
            img.astype(np.uint8),
            config=f"--psm 7 -c tessedit_char_whitelist={self.WHITELIST}",
        )
        return text.strip()
