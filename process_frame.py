import os
import cv2
import numpy as np

from draw_label_print import timestamp, draw_roi_poly, is_truck_txt
from is_truck import TruckDetector
from move_region_mask import RegionMasking
from segmentation import Segmenter
import global_params_variables

params = global_params_variables.ParamsDict()

model_path = params.get_value('model_path')
classes_path = params.get_value('classes_path')
colors_path = params.get_value('colors_path')
output_image_path = params.get_value('output_image_path')
segmentation_perc = params.get_value('segmentation_perc')


class FrameProcessor:
    def __init__(self, roi_comp):
        self.truck_detected_frame_index = None
        self.roi_comp = roi_comp
        self.motion_frames = {key: 0 for key in roi_comp.rois}
        self.motion_start_frame = {key: 0 for key in roi_comp.rois}
        self.truck_detected = False

    def process_frame(self, raw_frame, prev_frame, frame_height, frame_width, ts, back, frame_index):

        frames_dir = "saved_frames"
        os.makedirs(frames_dir, exist_ok=True)
        # timestamp(raw_frame, ts)

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
                self.truck_detected = True
                # is_truck_txt(raw_frame, roi_key, is_truck)
            else:
                self.truck_detected = False

            if self.truck_detected and frame_index % 30 == 0:  # write every 30 frames
                img_name = f"frame_{frame_index}.jpg"
                img_path = os.path.join(frames_dir, img_name)

                # Process segmentation
                # if  black is < 70 percent, save image for number prediction
                segmenter = Segmenter(model_path, classes_path, colors_path)
                pct, image = segmenter.process_images(raw_frame)
                #print(img_name, pct)
                if pct < 70:
                    cv2.imwrite(img_path, image)

        return raw_frame
