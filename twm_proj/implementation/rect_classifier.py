import numpy as np

from twm_proj.interface.rect_classifier import IRectClassifier, RectangleType


class RectClassifier(IRectClassifier):
    PLATE_SIZES = {
        RectangleType.ONE_ROW_PLATE: (520, 114),
        RectangleType.TWO_ROW_PLATE: (305, 214),
        RectangleType.MINI_PLATE: (315, 114),
    }

    PLATE_ASPECTS = {typ: size[0] / size[1] for typ, size in PLATE_SIZES.items()}

    EPSILON = 0.18

    def classify(self, rect: np.ndarray) -> RectangleType:
        typ, _ = self.classify_with_differences(rect)
        return typ

    def classify_with_differences(
        self, rect: np.ndarray
    ) -> tuple[RectangleType, dict[RectangleType, float]]:
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
