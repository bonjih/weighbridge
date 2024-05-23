import cv2
import numpy as np


def calculate_optical_flow(raw_frame, prev_frame, roi_points):
    """
    returns the direction of the object, i.e. out of gate or in gate
    """
    direction = ""

    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(np.array(roi_points))
    roi_gray_prev_frame = gray_prev_frame[y:y + h, x:x + w]
    roi_gray_raw_frame = gray_raw_frame[y:y + h, x:x + w]

    flow = cv2.calcOpticalFlowFarneback(roi_gray_prev_frame, roi_gray_raw_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
 
    # Check if the average magnitude exceeds the threshold
    direction = ""
    if magnitude.mean() > 0.09:  # Magnitude threshold for significant motion
        if magnitude.mean() > 0:  # Motion towards the right of the screen
            cv2.putText(raw_frame, "Truck Out", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            direction = "Truck Out"
        elif magnitude.mean() <= 0:  # Motion towards the top of the screen
            cv2.putText(raw_frame, "Truck In", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            direction = "Truck In"

    return raw_frame, direction
