import os

from ocr_e.ocr import OCR
from utils import group_predictions_sort
import global_params_variables

params = global_params_variables.ParamsDict()
output_roi_path = params.get_value('output_image_path')


def _ocr(raw_frame):
    ocr = OCR(detect=True)

    image_files = [os.path.join(output_roi_path, file) for file in os.listdir(output_roi_path) if file.endswith(
        ('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_path in image_files:
        text_list = ocr.predict([image_path])
        if text_list and text_list[0]:
            pass

        grouped_predictions = group_predictions_sort(text_list[-1])

        for group in grouped_predictions:
            predictions = [prediction[2] for prediction in group]

            # assume if group < 2, not a txt
            if any('.' in pred for pred in predictions) or len(predictions) < 2:
                continue

            concatenated_predictions = ''.join(predictions)
            print(concatenated_predictions)
            if concatenated_predictions[0].isalpha() and concatenated_predictions[-1].isdigit():
                return raw_frame, concatenated_predictions
