import cv2


class TruckDetector:
    """
    Identifies trucks in a reference frame based on threshold type.

    is_truck - if any pixels match the threshold in the ROI, a flag is raised True/False.
    """

    def __init__(self, thresh_type, w, h, roi_key, roi_is_truck, frame_index):
        self.thresh_type = thresh_type
        self.w = w
        self.h = h
        self.roi_key = roi_key
        self.roi_is_truck = roi_is_truck
        self.frame_index = frame_index

    def is_truck(self, frame_orig):
        thresh_resized = cv2.resize(self.thresh_type, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        current_frame_temp = frame_orig.copy()
        current_frame_temp[thresh_resized == 255] = (255, 255, 255)

        if len(current_frame_temp[thresh_resized == 255]) > 1000:
            cv2.putText(current_frame_temp, self.roi_key, (self.roi_is_truck[1][0] - 15, self.roi_is_truck[1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return current_frame_temp, True
        else:
            return frame_orig, False
