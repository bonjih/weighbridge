import os
import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

import global_params_variables

params = global_params_variables.ParamsDict()
confidence_thres = params.get_value('confidence_thres')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

processor = AutoImageProcessor.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")
model = AutoModelForImageClassification.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")


def predict_numbers(ROIs):
    predictions = []

    for roi_number, (coords, ROI) in ROIs.items():
        x, y, w, h = coords
        roi_pil = Image.fromarray(cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))

        inputs = processor(images=roi_pil, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(probabilities, dim=1)

        # Check if confidence is above the threshold
        if max_prob.item() > confidence_thres:
            predictions.append((x, y, predicted_class.item(), max_prob.item()))

    return predictions