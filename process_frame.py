import os
import cv2
import numpy as np

from hf_mnist.hf_mnist import predict_numbers
from object_direction import calculate_optical_flow
from ocr_main import _ocr
from process_truck import process_images_dir
from utils import group_predictions_sort, truck_number
from utils import timestamp, format_results, print_results, save_results
from is_truck import TruckDetector
from object_region_mask import RegionMasking
from segmentation import Segmenter
import global_params_variables

params = global_params_variables.ParamsDict()

model_path = params.get_value('model_path')
classes_path = params.get_value('classes_path')
colors_path = params.get_value('colors_path')
output_image_path = params.get_value('output_image_path')
segmentation_perc = params.get_value('segmentation_perc')
input_video_path = params.get_value('input_video_path')

directions = []
t_name = []


def func_call():
    ROIs = process_images_dir("saved_frames")
    print('Running Prediction')
    predictions = predict_numbers(ROIs)
    predictions = group_predictions_sort(predictions)
    # results of prediction
    print('Formatting / Saving Results')
    results = format_results(predictions)
    print_results(results)
    save_results(results[0], input_video_path, directions)


class FrameProcessor:
    def __init__(self, roi_comp):
        self.truck_detected_frame_index = None
        self.roi_comp = roi_comp
        self.motion_frames = {key: 0 for key in roi_comp.rois}
        self.motion_start_frame = {key: 0 for key in roi_comp.rois}
        self.truck_detected = False
        self.directions = {}
        self.ocr_results = {}

    def process_frame(self, raw_frame, prev_frame, frame_height, frame_width, ts, back, frame_index):
        os.makedirs(output_image_path, exist_ok=True)
        timestamp(raw_frame, ts)

        # ensure that self.truck_detected remains True if any ROI detects a truck
        truck_detected_in_frame = False

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()

            # draw_roi_poly(raw_frame, roi_key, roi_points)

            # Extract the bounding box of the ROI
            x, y, w, h = cv2.boundingRect(np.array(roi_points))
            roi_frame = raw_frame[y:y + h, x:x + w]
            prev_roi_frame = prev_frame[y:y + h, x:x + w]

            # Movement detection
            masking_instance = RegionMasking(roi_points)
            fgmask_1, fgmask_2, region = masking_instance.masking(prev_frame, raw_frame, back)
            thresh_type = masking_instance.threshold(fgmask_1, fgmask_2)

            # Process detection
            truck_instance = TruckDetector(thresh_type, frame_width, frame_height, roi_key, region, frame_index)
            frame_is_truck, is_truck = truck_instance.is_truck(raw_frame)

            if is_truck:
                truck_detected_in_frame = True
                # can use roi_key as truck in or out
                raw_frame, direction = calculate_optical_flow(raw_frame, prev_frame, roi_points)
                self.directions[frame_index] = direction  # get the direction for saving to audit csv

                if truck_detected_in_frame and frame_index % 30 == 0:  # write every n frames
                    img_name = f"frame_{frame_index}.jpg"
                    img_path = os.path.join(output_image_path, img_name)

                    # Process segmentation
                    # if black is < some percent, save image for number prediction
                    # get the best image for prediction
                    segmenter = Segmenter(model_path, classes_path, colors_path)
                    pct, image = segmenter.process_images(raw_frame)

                    if pct < segmentation_perc:
                        directions.append(list(self.directions.items())[0][1])
                        cv2.imwrite(img_path, image)
                        raw_frame, ocr_result = _ocr(raw_frame)
                        self.ocr_results[frame_index] = ocr_result
                        t_name.append(list(self.ocr_results.items())[0][1])

            if is_truck:
                truck_number(raw_frame, t_name)

        self.truck_detected = truck_detected_in_frame

        return raw_frame
