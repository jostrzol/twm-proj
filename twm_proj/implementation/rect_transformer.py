import numpy as np

from twm_proj.interface.rect_detector import SlantedRectangle
from twm_proj.interface.rect_transformer import IRectTransformer, RectangleImage


class RectTransformer(IRectTransformer):
    def transform(self, image: np.ndarray, rect: SlantedRectangle) -> RectangleImage:
        raise NotImplementedError()
