import os
import shutil
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

import global_params_variables

params = global_params_variables.ParamsDict()
output_audit_path = params.get_value('output_audit_path')

text_y = 25


def timestamp(frame, ts):
    cv2.putText(frame, f"TS(s): {ts / 1000:.3f}", (10, text_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


def truck_number(frame,  ocr_result):
    cv2.putText(frame, f" {ocr_result}", (125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


def draw_roi_poly(frame, roi_key, roi_points):
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 1)
    cv2.putText(frame, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def truck_direction(frame, direction):
    cv2.putText(frame, direction, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


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


def save_results(truck_number, video_name, direction):
    output_dir = os.path.dirname(output_audit_path)
    os.makedirs(output_dir, exist_ok=True)
    print(direction)
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
        "test_number": [test_number],
        "truck_direction": [direction[0]]
    }

    df = pd.DataFrame(data, columns=["date", "truck_number", "video_name", "test_number", "truck_direction"])
    df.to_csv(output_audit_path, index=False, mode='a', header=not os.path.exists(output_audit_path))


def group_predictions_sort(predictions):
    """Group predictions by similar x coordinates.
    Sort y from low to high (to get vertical numbers)
    """
    x_threshold = 10  # to group similar coordinates along x
    # Sort predictions by x-coordinate before grouping
    predictions.sort(key=lambda x: x[0])

    grouped_predictions = []
    current_group = []

    for predict in predictions:
        if not current_group:
            current_group.append(predict)
        else:
            last_x = current_group[-1][0]
            if abs(predict[0] - last_x) <= x_threshold:
                current_group.append(predict)
            else:
                grouped_predictions.append(current_group)
                current_group.sort(key=lambda x: x[1])
                current_group = [predict]

    if current_group:
        current_group.sort(key=lambda x: x[1])  # sort final group by y, low to high
        grouped_predictions.append(current_group)

    return grouped_predictions


def clear_directory(directory_path):
    # clear img_roi dir
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
