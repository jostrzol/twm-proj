from subprocess import run

import cv2
import numpy as np
import seaborn_image as isns
from matplotlib import pyplot as plt
import os


def show(image: np.ndarray, height: float = 6):
    [img_height, img_width, *_] = image.shape
    aspect = img_width / img_height
    size = np.array([height * aspect, height])
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    if len(image.shape) == 2:
        ax.imshow(image, aspect="auto", cmap="gray")
    else:
        ax.imshow(image[..., ::-1], aspect="auto", cmap="hsv")
    ax.axis("off")
    plt.show()


def show_contours(image: np.ndarray, contours: np.ndarray, height: float = 6):
    contours_image = cv2.drawContours(np.copy(image), contours, -1, (0, 0, 255), 2)
    show(contours_image, height=height)


def show_collage(images: list[np.ndarray], col_wrap: int = 5, height: float = 3):
    if len(images) == 0:
        return

    if len(images[0].shape) == 2:
        cmap = "gray"
    else:
        cmap = "hsv"
        images = [img[..., ::-1] for img in images]

    isns.ImageGrid(images, col_wrap=col_wrap, cbar=False, cmap=cmap, height=height)


def git_root() -> str:
    result = run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    return result.stdout.strip()


def cd_git_root():
    os.chdir(git_root())
