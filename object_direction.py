import cv2
import numpy as np

def calculate_optical_flow(raw_frame, prev_frame, roi_points):
    """
    Returns the direction of the object, i.e., out of gate or in gate.
    """
    direction = ""

    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(np.array(roi_points))
    roi_gray_prev_frame = gray_prev_frame[y:y + h, x:x + w]
    roi_gray_raw_frame = gray_raw_frame[y:y + h, x:x + w]

    flow = cv2.calcOpticalFlowFarneback(roi_gray_prev_frame, roi_gray_raw_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mean_flow_x = np.mean(flow[..., 0])
    mean_flow_y = np.mean(flow[..., 1])

    # Determine direction based on the average flow vector components
    if magnitude.mean() > 0.09:  # Magnitude threshold for significant motion
        if mean_flow_x > 0:  # Motion to the right
            direction = "Truck Out"
        elif mean_flow_x < 0:  # Motion to the left
            direction = "Truck In"
        elif mean_flow_y > 0:  # Motion downward
            direction = "Truck Down"
        elif mean_flow_y < 0:  # Motion upward
            direction = "Truck Up"
    
    # Annotate the frame with the direction
    cv2.putText(raw_frame, direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Draw the motion vectors as arrows
    step_size = 16
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            start_point = (x + j, y + i)
            end_point = (int(x + j + flow[i, j, 0]), int(y + i + flow[i, j, 1]))
            cv2.arrowedLine(raw_frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.3)
    
    return raw_frame, direction

def truck_direction(frame, direction):
    # Function to display direction on the frame
    pass  # Implement as needed
