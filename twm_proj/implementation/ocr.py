import numpy as np

from twm_proj.interface.ocr import IOcr


class Ocr(IOcr):
    def scan_text(self, image: np.ndarray) -> str:
        raise NotImplementedError()
