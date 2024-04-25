from argparse import ArgumentParser, FileType
from twm_proj.implementation.contour_detector import ContourDetector
from twm_proj.implementation.edge_filter import EdgeFilter
from twm_proj.implementation.image_reader import ImageReader
from twm_proj.implementation.initial_filter import InitialFilter
from twm_proj.implementation.ocr import Ocr
from twm_proj.implementation.rect_classifier import RectClassifier
from twm_proj.implementation.rect_detector import RectDetector
from twm_proj.implementation.rect_transformer import RectTransformer

from twm_proj.licence_plate_detector import LicencePlateDetector


def main():
    parser = ArgumentParser()
    parser.add_argument("files", type=FileType("rb"), nargs="+")
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
    )

    for file in args.files:
        plates = [*detector.detect(file)]
        print([plate.text for plate in plates])


if __name__ == "__main__":
    main()
