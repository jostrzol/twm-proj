import cv2
import numpy as np

from twm_proj.interface.edge_filter import IEdgeFilter


class EdgeFilter(IEdgeFilter):
    def filter(self, image: np.ndarray) -> np.ndarray:
        image = cv2.edgePreservingFilter(image, flags=1, sigma_s=30, sigma_r=0.15)
        image = cv2.Canny(image, 20, 100)
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

    @staticmethod
    def _vert_line_kernel(side: int, width: int) -> np.ndarray:
        kernel = np.zeros((side, side), np.uint8)
        center = side // 2
        left = center - width // 2
        right = left + width
        kernel[:, left:right] = 255
        return kernel

    @staticmethod
    def _horiz_line_kernel(side: int, width: int) -> np.ndarray:
        kernel = np.zeros((side, side), np.uint8)
        center = side // 2
        top = center - width // 2
        bottom = top + width
        kernel[top:bottom, :] = 255
        return kernel

    @staticmethod
    def _ring_kernel(side: int, r: float, R: float) -> np.ndarray:
        numspace_01 = np.linspace(0, 1, side)
        xs, ys = np.meshgrid(numspace_01, numspace_01)
        points = np.vstack([xs.ravel(), ys.ravel()]).T.reshape(side, side, 2)
        center = np.array([0.5, 0.5])
        dists = np.sqrt(np.square(points - center).sum(axis=2))
        kernel = np.zeros_like(dists, np.uint8)
        mask = np.logical_and(dists >= r, dists <= R)
        kernel[mask] = 255
        return kernel

    @staticmethod
    def _edge_kernel(side: int, width: int) -> np.ndarray:
        kernel = np.zeros((side, side), np.uint8)
        kernel[:, :-width:-1] = 255
        return kernel

    @staticmethod
    def _square_kernel(side: int) -> np.ndarray:
        return np.ones((side, side), np.uint8)
