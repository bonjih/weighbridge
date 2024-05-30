import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from object_direction import calculate_optical_flow
from ocr_main import _ocr
from utils import truck_number, truck_direction, draw_roi_poly
from utils import timestamp, save_results
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
first_frame_index_list = []


# crop to get a zoom
def crop(img, roi_key):
    height, width, _ = img.shape
    half_height = height // 2

    if roi_key == 'roi_2':
        left_half_image = img[:, :width // 2]
        top_half_image = left_half_image[:half_height, :]

    else:
        right_half_image = img[:, width // 2:]
        top_half_image = right_half_image[:half_height, :]

    return top_half_image


# TODO move to utils
def write_results(image, frame_index, raw_frame):
        img_name = f"frame_{frame_index}.jpg"
        img_path = os.path.join(output_image_path, img_name)
        cv2.imwrite(img_path, image)
        _, ocr_result = _ocr(raw_frame)
        t_name.append(ocr_result)
        save_results(t_name, input_video_path, direction_list)


class FrameProcessor:
    def __init__(self, roi_comp):
        self.roi_comp = roi_comp
        self.truck_detected = False
        self.model_path = model_path
        self.classes_path = classes_path
        self.colors_path = colors_path
        self.output_image_path = output_image_path
        os.makedirs(self.output_image_path, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.segmentation_future = None

    def stop_segmentation(self):
        if self.segmentation_future is not None:
            self.segmentation_future.cancel()

    def segment_image(self, raw_frame, frame_index, roi_key):
        segmenter = Segmenter(self.model_path, self.classes_path, self.colors_path)
        pct, image = segmenter.process_images(raw_frame)
        image = crop(image, roi_key)

        return pct, image

    def process_frame(self, raw_frame, prev_frame, frame_height, frame_width, ts, back, frame_index):
        timestamp(raw_frame, ts)
        truck_detected_in_frame = False
        img = None
        pct = 0

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            x, y, w, h = cv2.boundingRect(np.array(roi_points))
            roi_frame = raw_frame[y:y + h, x:x + w]
            prev_roi_frame = prev_frame[y:y + h, x:x + w]

            # draw_roi_poly(raw_frame, roi_key, roi_points)

            masking_instance = RegionMasking(roi_points)
            fg_mask_1, fg_mask_2, region = masking_instance.masking(prev_frame, raw_frame, back)
            thresh_type = masking_instance.threshold(fg_mask_1, fg_mask_2)

            truck_instance = TruckDetector(thresh_type, frame_width, frame_height, roi_key, region, frame_index)
            frame_is_truck, is_truck = truck_instance.is_truck(raw_frame)

            if is_truck and frame_index > 1:
                truck_detected_in_frame = True

                if roi_key == 'roi_1':
                    direction = "Truck Out"
                else:
                    direction = "Truck In"

                truck_direction(raw_frame, direction)  # to put direction to video
                direction_list.append(direction)

                # store the index of the first frame where a truck is detected
                if not first_frame_index_list:
                    first_frame_index_list.append(frame_index)

                # a frame delay from when the truck is at the gate (ROI) th haulage, reduces unnecessary segmentation
                if roi_key == 'roi_1':
                    truck_wait = first_frame_index_list[0] + 600
                else:
                    truck_wait = first_frame_index_list[0] + 300

                if frame_index % 30 == 0:
                    if frame_index > truck_wait:
                        future = self.executor.submit(self.segment_image, raw_frame, frame_index, roi_key)
                        pct, img = future.result()
                        pct_list.append(pct)
                        pct_list.sort()

            if img is not None and not self.truck_detected:
                if roi_key == 'roi_1' and pct < 70:
                    write_results(img, frame_index, raw_frame)
                elif roi_key == 'roi_2' and pct < 91:
                    write_results(img, frame_index, raw_frame)

            truck_number(raw_frame, t_name)

        # stop the segmentation thread but don't reset the first detected frame index when truck is not in ROI
        if not truck_detected_in_frame:
            self.stop_segmentation()
            return raw_frame

        self.truck_detected = truck_detected_in_frame
        return raw_frame
