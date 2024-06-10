import cv2
import numpy as np

from twm_proj.interface.edge_filter import IEdgeFilter


class EdgeFilter(IEdgeFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        image = cv2.edgePreservingFilter(image, flags=1, sigma_s=30, sigma_r=0.15)
        image = cv2.Canny(image, 20, 100)
        return image
