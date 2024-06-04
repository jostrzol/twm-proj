import cv2
import numpy as np

from matplotlib import pyplot as plt
import seaborn_image as isns


def show(image: np.ndarray):
    [height, width, *_] = image.shape
    aspect = width / height
    size = np.array([aspect, 1])
    size = size / size.max() * 4
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    if len(image.shape) == 2:
        ax.imshow(image, aspect="auto", cmap="gray")
    else:
        ax.imshow(image[..., ::-1], aspect="auto", cmap="hsv")
    ax.axis("off")
    plt.show()


def show_contours(image: np.ndarray, contours: np.ndarray):
    contours_image = cv2.drawContours(np.copy(image), contours, -1, (0, 0, 255), 2)
    show(contours_image)


def show_collage(images: list[np.ndarray], col_wrap: int = 5, height: float = 3):
    reversed = [img[..., ::-1] for img in images]
    isns.ImageGrid(reversed, col_wrap=col_wrap, cbar=False, cmap="hsv", height=height)