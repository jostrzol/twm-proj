import cv2
import numpy as np

from twm_proj.interface.rect_classifier import IRectClassifier, RectangleType


class RectClassifier(IRectClassifier):
    PLATE_SIZES = {
        RectangleType.ONE_ROW_PLATE: (520, 114),
        RectangleType.TWO_ROW_PLATE: (305, 214),
        RectangleType.MINI_PLATE: (315, 114),
    }

    PLATE_ASPECTS = {typ: size[0] / size[1] for typ, size in PLATE_SIZES.items()}

    EPSILON = 0.24

    def classify(self, rect: np.ndarray) -> RectangleType:
        typ, _ = self.classify_with_differences(rect)
        return typ

    def classify_with_differences(
        self, rect: np.ndarray
    ) -> tuple[RectangleType, dict[RectangleType, float]]:
        hsv_rect = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, k_means = cv2.kmeans(
            hsv_rect.astype(np.float32).reshape((-1, 3)),
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        k_means = np.sort(k_means, axis=0)
        if (k_means[1] - k_means[0] < np.array([0, 0, 80])).any():
            return RectangleType.NOT_PLATE, {}

        [height, width, *_] = rect.shape
        aspect = width / height
        relative_differences = {
            typ: np.abs(plate_aspect - aspect) / plate_aspect
            for typ, plate_aspect in self.PLATE_ASPECTS.items()
        }
        best_type, best_difference = min(
            relative_differences.items(), key=lambda pair: pair[1]
        )
        typ = best_type if best_difference <= self.EPSILON else RectangleType.NOT_PLATE
        return typ, relative_differences
