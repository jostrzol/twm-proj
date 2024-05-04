import dataclasses
import json
import sys

import numpy as np

from argparse import ArgumentParser
from twm_proj.implementation.contour_detector import ContourDetector
from twm_proj.implementation.edge_filter import EdgeFilter
from twm_proj.implementation.image_reader import ImageReader
from twm_proj.implementation.initial_filter import InitialFilter
from twm_proj.implementation.ocr import Ocr
from twm_proj.implementation.pre_ocr import PreOcr
from twm_proj.implementation.rect_classifier import RectClassifier
from twm_proj.implementation.rect_detector import RectDetector
from twm_proj.implementation.rect_transformer import RectTransformer

from twm_proj.licence_plate_detector import LicencePlateDetector, LicencePlate


def main():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    detector = LicencePlateDetector(
        image_reader=ImageReader(),
        initial_filter=InitialFilter(),
        edge_filter=EdgeFilter(),
        contour_detector=ContourDetector(),
        rect_detector=RectDetector(),
        rect_transformer=RectTransformer(),
        rect_classifier=RectClassifier(),
        ocr=Ocr(),
        pre_ocr=PreOcr()
    )

    for path in args.files:
        with open(path, "rb") as file:
            plates = [*detector.detect(file)]
        result = {"file": path, "plates": plates}
        json.dump(result, sys.stdout, cls=CustomJSONEncoder)
        print()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, LicencePlate):
            dict = dataclasses.asdict(o)
            dict.pop("image")
            return dict
        elif dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


if __name__ == "__main__":
    main()
