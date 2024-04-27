from typing import Any, BinaryIO, Generator
from twm_proj.interface.rect_detector import IRectDetector
from twm_proj.interface.edge_filter import IEdgeFilter
from twm_proj.interface.contour_detector import IContourDetector
from twm_proj.interface.image_reader import IImageReader
from twm_proj.interface.initial_filter import IInitialFilter
from twm_proj.interface.ocr import IOcr
from twm_proj.interface.rect_classifier import IRectClassifier, RectangleType
from twm_proj.interface.rect_transformer import IRectTransformer

from dataclasses import dataclass

import numpy as np


@dataclass
class LicencePlate:
    contour: np.ndarray
    image: np.ndarray
    text: str


class LicencePlateDetector:
    def __init__(
        self,
        image_reader: IImageReader,
        initial_filter: IInitialFilter,
        edge_filter: IEdgeFilter,
        contour_detector: IContourDetector,
        rect_detector: IRectDetector,
        rect_transformer: IRectTransformer,
        rect_classifier: IRectClassifier,
        ocr: IOcr,
    ):
        self._image_reader = image_reader
        self._initial_filter = initial_filter
        self._edge_filter = edge_filter
        self._contour_detector = contour_detector
        self._rect_detector = rect_detector
        self._rect_transformer = rect_transformer
        self._rect_classifier = rect_classifier
        self._ocr = ocr

    def detect(self, image_file: BinaryIO) -> Generator[LicencePlate, Any, None]:
        image = self._image_reader.read(image_file)
        filtered = self._initial_filter.filter(image)
        edges = self._edge_filter.filter(filtered)
        for contour in self._contour_detector.detect(edges):
            rect = self._rect_detector.detect(contour)
            if rect is None:
                continue
            rect_image = self._rect_transformer.transform(image, rect)
            rect_type = self._rect_classifier.classify(rect_image)
            if rect_type == RectangleType.NOT_PLATE:
                continue
            text = self._ocr.scan_text(rect_image)
            yield LicencePlate(contour=contour, image=rect_image, text=text)
