import os
import cv2

x_threshold = 10  # to group similar coordinates along x


def process_images_dir(directory):
    """Process all images in the given directory using the thresholding function."""
    all_ROIs = {}

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            ROIs = thresholding(image_path)
            all_ROIs.update(ROIs)
        else:
            print("Skipping file:", filename)

    return all_ROIs


def thresholding(img):
    image = cv2.imread(img)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 200, 255)
    # cv2.imshow("Output", canny)
    # cv2.waitKey(0)

    ROI_number = 0
    ROIs = {}  # store ROIs and their coordinates
    roi_dir = "img_roi"
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi_area = w * h

        if 150 < roi_area < 2000:
            ROI = original[y:y + h, x:x + w]
            ROIs[ROI_number] = ((x, y, w, h), ROI)  # Store ROI coordinates and image array

            roi_filename = os.path.join(roi_dir, f"roi_{ROI_number}'_'{x}'_'{y}.png")
            cv2.imwrite(roi_filename, ROI)

        ROI_number += 1

    return ROIs


def group_predictions_sort(predictions):
    """Group predictions by similar x coordinates.
    Sort y from low to high (to get vertical numbers)
    """
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
                current_group.sort(key=lambda x: x[1])  # sort final group by y, low to high
                current_group = [predict]

    if current_group:
        current_group.sort(key=lambda x: x[1])  # sort final group by y, low to high
        grouped_predictions.append(current_group)

    return grouped_predictions
