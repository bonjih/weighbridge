
import cv2

text_y = 25


def timestamp(frame, ts):
    cv2.putText(frame, f"TS(s): {ts / 1000:.3f}", (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)


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


def print_results(predictions):
    for group in predictions:
        if len(group) > 5:  # assume trucks have a number > 5
            group_str = ' '.join([f"{number} ({x}, {y}) {confidence:.3f}" for x, y, number, confidence in group])
            print(group_str)
            print()
            print([number for _, _, number, _ in group])
