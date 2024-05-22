import numpy as np
import cv2


def calculate_black_pixel_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray == 0)
    total_pixels = gray.size
    pct = (black_pixels / total_pixels) * 100
    return pct


class Segmenter:
    def __init__(self, model_path, classes_path, colors_path):
        self.model_path = model_path
        self.classes_path = classes_path
        self.colors_path = colors_path
        self.CLASSES = self.load_classes()
        self.COLORS = self.load_colors()
        self.net = self.load_model()

    def load_classes(self):
        with open(self.classes_path) as f:
            return f.read().strip().split("\n")

    def load_colors(self):
        if self.colors_path:
            with open(self.colors_path) as f:
                COLORS = f.read().strip().split("\n")
            COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
            return np.array(COLORS, dtype="uint8")
        else:
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(self.CLASSES) - 1, 3), dtype="uint8")
            return np.vstack([[0, 0, 0], COLORS]).astype("uint8")

    def load_model(self):
        return cv2.dnn.readNet(self.model_path)

    def perform_segmentation(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 256), 0, swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward()

    def extract_truck(self, image, class_map):
        mask = (class_map == self.CLASSES.index("Truck")).astype("uint8") * 255
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(image, mask_rgb)

    def process_images(self, image):
        output = self.perform_segmentation(image)

        class_map = np.argmax(output[0], axis=0)
        image = self.extract_truck(image, class_map)
        pct = calculate_black_pixel_percentage(image)
        return pct, image
