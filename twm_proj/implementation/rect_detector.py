from typing import Any

from twm_proj.interface.rect_detector import IRectDetector, SlantedRectangle


class RectDetector(IRectDetector):
    # TODO: specify contour type
    def detect(self, contour: Any) -> SlantedRectangle | None:
        raise NotImplementedError()
