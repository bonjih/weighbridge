import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from object_direction import calculate_optical_flow
from ocr_main import _ocr
from process_truck import process_images_dir
from utils import group_predictions_sort, truck_number, truck_direction, draw_roi_poly
from utils import timestamp, format_results, save_results
from is_truck import TruckDetector
from object_region_mask import RegionMasking
from segmentation import Segmenter
import global_params_variables

params = global_params_variables.ParamsDict()

model_path = params.get_value('model_path')
classes_path = params.get_value('classes_path')
colors_path = params.get_value('colors_path')
output_image_path = params.get_value('output_image_path')
input_video_path = params.get_value('input_video_path')

direction_list = []
t_name = []
pct_list = []


def func_call():
    ROIs = process_images_dir("saved_frames")
    print('Running Prediction')
    predictions = predict_numbers(ROIs)
    predictions = group_predictions_sort(predictions)
    print('Formatting / Saving Results')
    try:
        results = format_results(predictions)
        save_results(results[0], input_video_path, direction_list)
    except Exception as e:
        print('No results generated:', e)


class FrameProcessor:
    def __init__(self, roi_comp):
        self.roi_comp = roi_comp
        self.truck_detected = False
        self.directions = {}
        self.ocr_results = {}
        self.model_path = model_path
        self.classes_path = classes_path
        self.colors_path = colors_path
        self.output_image_path = output_image_path
        os.makedirs(self.output_image_path, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def segment_image(self, raw_frame, frame_index, roi_key):  # to segment while streaming video
        segmenter = Segmenter(self.model_path, self.classes_path, self.colors_path)

        pct, image = segmenter.process_images(raw_frame)
        pct_list.append(pct)

        pct_list.sort()

        if pct_list:
            if pct <= min(pct_list):  # get the min pct, min is the img to process
                img_name = f"frame_{frame_index}.jpg"
                img_path = os.path.join(self.output_image_path, img_name)
                cv2.imwrite(img_path, image)
                _, ocr_result = _ocr(raw_frame)
                self.ocr_results[frame_index] = ocr_result

                t_name.append(ocr_result)
                direction_list.append(self.directions[frame_index])
                return raw_frame

    def process_frame(self, raw_frame, prev_frame, frame_height, frame_width, ts, back, frame_index):
        timestamp(raw_frame, ts)
        truck_detected_in_frame = False

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            x, y, w, h = cv2.boundingRect(np.array(roi_points))
            roi_frame = raw_frame[y:y + h, x:x + w]
            prev_roi_frame = prev_frame[y:y + h, x:x + w]

            # draw_roi_poly(raw_frame, roi_key, roi_points)

            masking_instance = RegionMasking(roi_points)
            fgmask_1, fgmask_2, region = masking_instance.masking(prev_frame, raw_frame, back)
            thresh_type = masking_instance.threshold(fgmask_1, fgmask_2)

            truck_instance = TruckDetector(thresh_type, frame_width, frame_height, roi_key, region, frame_index)
            frame_is_truck, is_truck = truck_instance.is_truck(raw_frame)

            if is_truck:
                truck_detected_in_frame = True

                if roi_key == 'roi_1':  # Motion towards the right of the screen
                    direction = "Truck Out: "
                    truck_direction(raw_frame, direction)
                elif roi_key == 'roi_2':  # Motion towards the top of the screen
                    direction = "Truck In: "
                    truck_direction(raw_frame, direction)

                    self.directions[frame_index] = direction

                if frame_index % 30 == 0:
                    self.executor.submit(self.segment_image, raw_frame, frame_index, roi_key)

                truck_number(raw_frame, t_name)

        self.truck_detected = truck_detected_in_frame

        return raw_frame
