import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from twm_proj.interface.ocr import IOcr


class Ocr(IOcr):

    def __init__(self, model_path="../models/ocr.keras"):
        model = tf.keras.models.load_model(model_path)
        self._model = model

    def scan_text(self, letters: list[np.ndarray]) -> str:
        predicted_text = ""
        for letter in letters:
            resized_image = cv2.resize(letter, (60, 80))
            img = np.expand_dims(resized_image, axis=0)

            predictions = self._model.predict(img, verbose=0)

            predicted_class_index = np.argmax(predictions)

            def index_to_label(index):
                if 0 <= index < 26:
                    return chr(index + ord('A'))
                elif 26 <= index < 36:
                    return chr(index - 26 + ord('0'))
                else:
                    raise ValueError(f"Unsupported class index: {index}")

            predicted_label = index_to_label(predicted_class_index)
            predicted_text += predicted_label

        return predicted_text
