import cv2
import numpy as np

from twm_proj.interface.edge_filter import IEdgeFilter


class EdgeFilter(IEdgeFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        lower = np.quantile(image, 0.6)
        _, image = cv2.threshold(image, lower, 255, cv2.THRESH_BINARY)
        kernel_open = self._gaussian_kernel(3, 1)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)

        kernel_close = self._gaussian_kernel(8, 2)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
        return image

    @staticmethod
    def _gaussian_kernel(side: int, sigma: float) -> np.ndarray:
        ax = np.linspace(-(side - 1) / 2.0, (side - 1) / 2.0, side)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel_float = np.outer(gauss, gauss)
        kernel_norm = kernel_float / kernel_float.max()
        return (kernel_norm * 255).astype(np.uint8)

    @staticmethod
    def _circular_kernel(side: int) -> np.ndarray:
        numspace_01 = np.linspace(0, 1, side)
        xs, ys = np.meshgrid(numspace_01, numspace_01)
        points = np.vstack([xs.ravel(), ys.ravel()]).T.reshape(side, side, 2)
        center = np.array([0.5, 0.5])
        dists = np.sqrt(np.square(points - center).sum(axis=2))
        kernel_norm = -dists / dists.max() + 1
        return (kernel_norm * 255).astype(np.uint8)
