from subprocess import run

import cv2
import numpy as np
import seaborn_image as isns
from matplotlib import pyplot as plt
import os
from typing import Iterable
import matplotlib


def show(image: np.ndarray, height: float = 30):
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


def show_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    height: float = 30,
    n_colors: int = 8,
    thickness: int = 3,
):
    image = np.copy(image)
    colors = matplotlib.cm.Set1(range(n_colors))
    contour_by_color = [
        (color, contours[i::n_colors]) for i, color in enumerate(colors)
    ]
    for color, contour_group in contour_by_color:
        color_uint = tuple(map(int, color[2::-1] * 255))
        image = cv2.drawContours(image, contour_group, -1, color_uint, thickness)
    show(image, height=height)


def show_collage(
    images: list[np.ndarray],
    col_wrap: int = 5,
    height: float = 3,
    texts: Iterable[str] | None = [],
):
    if len(images) == 0:
        return

    color_images = []
    for image in images:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color_images.append(image)

    rgb_images = [img[..., ::-1] for img in color_images]
    grid = isns.ImageGrid(rgb_images, col_wrap=col_wrap, cbar=False, height=height)

    if not texts:
        texts = []
    for axes, text in zip(grid.axes.flat, texts):
        axes.set_title(text)


def git_root() -> str:
    result = run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    return result.stdout.strip()


def cd_git_root():
    os.chdir(git_root())


def filter_by_color(
    contours: list[np.ndarray], *color_indexes: list[int], n_colors=8
) -> list[np.ndarray]:
    for index in color_indexes:
        contours = contours[index::n_colors]
    return contours
