from typing import Any, BinaryIO, Generator
from twm_proj.interface.rect_detector import IRectDetector
from twm_proj.interface.edge_filter import IEdgeFilter
from twm_proj.interface.contour_detector import IContourDetector
from twm_proj.interface.image_reader import IImageReader
from twm_proj.interface.initial_filter import IInitialFilter
from twm_proj.interface.ocr import IOcr
from twm_proj.interface.pre_ocr import IPreOcr
from twm_proj.interface.rect_classifier import IRectClassifier, RectangleType
from twm_proj.interface.rect_transformer import IRectTransformer

from dataclasses import dataclass

import numpy as np


@dataclass
class LicencePlate:
    text: str
    rect: np.ndarray
    image: np.ndarray


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
        pre_ocr: IPreOcr,
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
        self._pre_ocr = pre_ocr

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
            ocr_image_cut = self._pre_ocr.cut(rect_image)
            ocr_image_grayscale = self._pre_ocr.to_grayscale(ocr_image_cut)
            ocr_image_filtered = self._pre_ocr.filter_by_size(ocr_image_grayscale)
            text = self._ocr.scan_text(ocr_image_filtered)
            yield LicencePlate(rect=rect, image=rect_image, text=text)
