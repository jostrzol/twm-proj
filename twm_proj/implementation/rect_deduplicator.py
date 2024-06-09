import numpy as np

from twm_proj.interface.rect_deduplicator import IRectDeduplicator

SAME_RECT_MAX_DIFF = 25


class RectDeduplicator(IRectDeduplicator):

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous_rects: list[np.ndarray] = []

    def is_dupe(self, rect: np.ndarray) -> bool:
        for prev_rect in self._previous_rects:
            diff_abs = np.abs(prev_rect - rect).sum()
            if diff_abs < SAME_RECT_MAX_DIFF:
                return True

        self._previous_rects.append(rect)
        return False
