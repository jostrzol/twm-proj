import cv2
import numpy as np

from twm_proj.interface.edge_filter import IEdgeFilter


class EdgeFilter(IEdgeFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(image, 100, 200)

        kernel = np.ones((10, 10), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges_closed
