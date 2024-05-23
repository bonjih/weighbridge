import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

import global_params_variables

params = global_params_variables.ParamsDict()
output_audit_path = params.get_value('output_audit_path')

text_y = 25


def timestamp(frame, ts):
    cv2.putText(frame, f"TS(s): {ts / 1000:.3f}", (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)


def draw_optical_flow(raw_frame, magnitude, x, w, y, h, mean_angle):
    # Draw the direction arrow
    if magnitude.mean() > 1:  # Threshold to consider significant motion
        end_point_x = int(x + w / 2 + 10 * np.cos(mean_angle))
        end_point_y = int(y + h / 2 + 10 * np.sin(mean_angle))
        cv2.arrowedLine(raw_frame, (x + w // 2, y + h // 2), (end_point_x, end_point_y), (0, 255, 0), 2)


def draw_roi_poly(frame, roi_key, roi_points):
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 1)
    cv2.putText(frame, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def is_truck_txt(frame, roi_key, is_truck):
    if is_truck:
        is_truck = 'Truck - Writing frames'
    else:
        is_truck = 'No Truck'
    cv2.putText(frame, f"{roi_key}: {is_truck}", (10, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)


def format_results(predictions):
    for group in predictions:
        if len(group) > 5:  # assume trucks have a number > 5
            # the entire result with conf level and x/y of roi
            group_str = ' '.join([f"{number} ({x}, {y}) {confidence:.3f}" for x, y, number, confidence in group])
            trunck_num = [number for _, _, number, _ in group]
            return [trunck_num, group_str]


def print_results(results):
    print(results[0])
    print()
    print(results[1])


def save_results(truck_number, video_name):
    output_dir = os.path.dirname(output_audit_path)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_audit_path):
        # Read the existing CSV file to find the latest test number
        df_existing = pd.read_csv(output_audit_path)
        if not df_existing.empty and 'test_number' in df_existing.columns:
            last_test_number = df_existing['test_number'].max()
        else:
            last_test_number = 0
    else:
        last_test_number = 0

    test_number = last_test_number + 1
    test_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "date": [test_time],
        "truck_number": [truck_number],
        "video_name": [video_name],  # file name, assuming test vids are mp4's
        "test_number": [test_number]
    }

    df = pd.DataFrame(data, columns=["date", "truck_number", "video_name", "test_number"])

    df.to_csv(output_audit_path, index=False, mode='a', header=not os.path.exists(output_audit_path))


