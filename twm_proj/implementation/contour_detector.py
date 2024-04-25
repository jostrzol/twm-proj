from typing import Any, Generator

import numpy as np

from twm_proj.interface.contour_detector import IContourDetector


class ContourDetector(IContourDetector):
    # TODO: specify return type
    def detect(self, image: np.ndarray) -> Generator[Any, Any, None]:
        raise NotImplementedError()
