import os
import cv2

import global_params_variables

params = global_params_variables.ParamsDict()
output_roi_path = params.get_value('output_roi_path')


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

    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi_area = w * h

        if 150 < roi_area < 2000:
            ROI = original[y:y + h, x:x + w]
            ROIs[ROI_number] = ((x, y, w, h), ROI)  # Store ROI coordinates and image array

            roi_filename = os.path.join(output_roi_path, f"roi_{ROI_number}_{x}_{y}.png")
            cv2.imwrite(roi_filename, ROI)

        ROI_number += 1

    return ROIs

