import dataclasses
import json
import sys

import numpy as np

from argparse import ArgumentParser

from twm_proj.licence_plate_detector import LicencePlateDetector, LicencePlate
from twm_proj.interface.rect_classifier import RectangleType


def main():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    detector = LicencePlateDetector.default()

    for path in args.files:
        with open(path, "rb") as file:
            plates = [*detector.read_and_detect(file)]
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
        elif isinstance(o, RectangleType):
            return str(o).split(".")[-1]
        return super().default(o)


if __name__ == "__main__":
    main()
