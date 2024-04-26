import numpy as np

from twm_proj.interface.rect_transformer import IRectTransformer, RectangleImage


class RectTransformer(IRectTransformer):
    def transform(self, image: np.ndarray, rect: np.ndarray) -> RectangleImage:
        raise NotImplementedError()
