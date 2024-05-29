import torch
import cv2
import skimage
import numpy as np
from PIL import Image
import pathlib
from torch.utils.data import Dataset, DataLoader

current_path = pathlib.Path(__file__).parent.resolve()

torch.manual_seed(0)
np.random.seed(0)

# import related to parseq
from torchvision import transforms as T
from ocr_e.strhub.data.utils import Tokenizer
from ocr_e.strhub.models.utils import load_from_checkpoint
from ocr_e.craft_text_detector import (
    load_craftnet_model,
    get_prediction,
    export_detected_regions
)

import warnings

warnings.filterwarnings("ignore")


class ParseqDataset(Dataset):
    """

    Parseq Dataset loader

    Args:
        Dataset (list): List of Images
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        x = skimage.exposure.rescale_intensity(x, in_range='image', out_range='dtype')
        x = Image.fromarray(np.uint8(x)).convert('RGB')

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


class OCR:
    """OCR class
    """

    def __init__(self, detect=True,
                 eng_model_path=None,
                 detect_model_path=None,
                 enable_cuda=True,
                 batch_size=8,
                 text_threshold=0.5,
                 link_threshold=0.3,
                 low_text=0.3,
                 details=2,
                 lang=None,
                 fp16=False,
                 recognize_thres=0.5,
                 assume_straight_page=False) -> None:
        """

        OCR prediction initialisation

        Args:
            detect (bool, optional): To enable the text detection. Defaults to False.
            eng_model_path (_type_, optional): Path for english text recognition model. Defaults to None.
            detect_model_path (_type_, optional): Path for text detect model. Defaults to None.
            enable_cuda (bool, optional): To enable or disable cuda. Defaults to True.
            batch_size (int, optional): Prediction batch size for text recognition. Defaults to 8.
            text_threshold (float, optional): Text detection threshold to classify text or not. Defaults to 0.5.
            link_threshold (float, optional): To combine characters into words (distance). Defaults to 0.1.
            low_text (float, optional): Helps in padding while cropping results from text detection. Defaults to 0.3.
            details (int, optional): Output information controller. Defaults to 0.
            lang (list, optional): Text recognize language. Defaults to ["english"].
            fp16 (bool, optional): full precision vs half precision (experimental). Defaults to False.
            recognize_thres (float, optional): Threshold to filter the texts based on prediction confidence (text recognition). Defaults to 0.85.
            assume_straight_page (bool, optional): If True faster processing will be used, but this may cause errors for rotated text.
        """

        if lang is None:
            lang = ["english"]
        if enable_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.lang = lang
        self.fp16 = fp16
        self.recognize_thres = recognize_thres

        self.detect = detect
        self.batch_size = batch_size
        self.assume_straight_page = assume_straight_page

        if self.assume_straight_page:
            self.method = "rectangular"
        else:
            self.method = "crop"

        self.special_character = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '00', '01',
                                  '02', '03', '04', '05', '06', '07', '08', '09', ':', ';', '<', '=', '>', '?', '@',
                                  '[', '\\', ']', '^', '_', '`', '{', '|', '}']

        self.eng_model_path = eng_model_path
        self.detect_model_path = detect_model_path

        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text

        self.details = details
        self.load_model()

        if self.detect:
            if torch.cuda.is_available() and enable_cuda:  # load models
                self.gpu = True
                self.craft_net = load_craftnet_model(cuda=True, weight_path=self.detect_model_path,
                                                     half=self.fp16)
            else:
                self.gpu = False
                self.craft_net = load_craftnet_model(cuda=False, weight_path=self.detect_model_path)

    @staticmethod
    def get_transform():
        """Basic transform for prediction

        Returns:
            torch transforms: torch vision transformation
        """
        transforms = []

        transforms.extend([
            T.Resize([32, 128], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

    def load_model(self):
        """
        Load the required models into the memory
        """

        self.img_transform = self.get_transform()
        self.eng_character_set = """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
        self.eng_tokenizer = Tokenizer(self.eng_character_set)

        if self.fp16:
            self.eng_parseq = load_from_checkpoint("pretrained=parseq").to(self.device).half().eval()
        else:
            self.eng_parseq = load_from_checkpoint("pretrained=parseq").to(self.device).eval()

    @staticmethod
    def sort_bboxes(contours):

        c = np.array(contours)
        max_height = np.median(c[::, 3]) * 0.5

        # Sort the contours by y-value
        by_y = sorted(contours, key=lambda k: k[1])  # y values

        line_y = by_y[0][1]  # first y
        line = 1
        by_line = []

        # Assign a line number to each contour
        for x, y, w, h in by_y:
            if y > line_y + max_height:
                line_y = y
                line += 1

            by_line.append((line, x, y, w, h))

        # This will now sort automatically by line then by x
        contours_sorted = [[x, y, w, h] for line, x, y, w, h in sorted(by_line)]
        line_info = [line for line, x, y, w, h in sorted(by_line)]

        return contours_sorted, line_info

    def craft_detect(self, image, **kwargs):
        """Text detection predict

        Args:
            image (numpy array): image numpy array

        Returns:
            list: list of cropped numpy arrays for text detected
            list: Bbox information
        """
        size = max(image.shape[0], image.shape[1], 640)

        # Reshaping to the nearest size
        size = min(size, 2560)

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.gpu,
            long_size=size,
            poly=False,
            half=self.fp16
        )

        new_bbox = []

        for bb in prediction_result:
            xs = bb[:, 0]
            ys = bb[:, 1]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            x, y, w, h = min_x, min_y, max_x - min_x, max_y - min_y
            if w > 0 and h > 0:
                new_bbox.append([x, y, w, h])

        if len(new_bbox):
            ordered_new_bbox, line_info = self.sort_bboxes(new_bbox)

            updated_prediction_result = []
            for ordered_bbox in ordered_new_bbox:
                index_val = new_bbox.index(ordered_bbox)
                updated_prediction_result.append(prediction_result[index_val])

            # export detected text regions
            exported_file_paths = export_detected_regions(
                image=image,
                regions=updated_prediction_result,  # ["boxes"],
                method=self.method
            )

            updated_prediction_result = [(i, line) for i, line in zip(updated_prediction_result, line_info)]

        else:
            updated_prediction_result = []
            exported_file_paths = []

        torch.cuda.empty_cache()

        return exported_file_paths, updated_prediction_result

    @staticmethod
    def read_image_input(image):
        """Reads the input image

        Args:
            image: Path, bytes and numpy array

        Returns:
            numpy array: image numpy array
        """
        if type(image) == str:
            img = cv2.imread(image)

        elif type(image) == bytes:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elif type(image) == np.ndarray:
            if len(image.shape) == 2:  # grayscale
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                img = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return img

    def text_recognize_batch(self, exported_regions):
        """Text recognition predictor

        Args:
            exported_regions (list): list of numpy array

        Returns:
            list: list of predicted text and confidence information
        """

        dataset = ParseqDataset(exported_regions, transform=self.img_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        eng_label_list = []
        eng_confidence_list = []

        for data in dataloader:
            if self.fp16:
                data = data.to(self.device).half()
            else:
                data = data.to(self.device)

            # english prediction
            if "english" in self.lang:
                with torch.cuda.amp.autocast() and torch.inference_mode():
                    logits = self.eng_parseq(data)
                # Greedy decoding
                pred = logits.softmax(-1)
                eng_preds, eng_confidence = self.eng_tokenizer.decode(pred)
                eng_label_list.extend(eng_preds)
                eng_confidence_list.extend(eng_confidence)

        text_list = []
        conf_list = []

        for e_l, e_c in zip(eng_label_list, eng_confidence_list):
            eng_conf = torch.mean(e_c)
            eng_conf = eng_conf.detach().cpu().numpy().item()

            if eng_conf >= self.recognize_thres:
                text_list.append(e_l)
                conf_list.append(eng_conf)
            else:
                text_list.append("")
                conf_list.append(0.0)

        torch.cuda.empty_cache()

        return text_list, conf_list

    def output_formatter(self, text_list, conf_list, updated_prediction_result=None):
        """Output structure formatter

        Args:
            text_list (list): text information
            conf_list (list): confidence information
            updated_prediction_result (list, optional): bbox information. Defaults to None.

        Returns:
            list: output results
        """
        final_result = []

        if not self.details:
            for text in text_list:
                final_result.append(text)

        elif self.details == 1:
            for text, conf in zip(text_list, conf_list):
                final_result.append((text, conf))

        elif self.details == 2 and updated_prediction_result is not None:
            for text, conf, bbox in zip(text_list, conf_list, updated_prediction_result):
                vertices_array, _ = bbox
                x, y = vertices_array[0]
                area = cv2.contourArea(vertices_array)

                if area > 300:
                    final_result.append((x, y, text, conf))

        elif self.details == 2 and updated_prediction_result is None:
            for text, conf in zip(text_list, conf_list):
                final_result.append((text, conf))

        return final_result

    def predict(self, image):

        """ Detect and recognize text information

        Returns:
            List: extracted text information
        """

        # To handle multiple images
        if isinstance(image, list):
            text_list = []

            if self.detect:
                for img in image:
                    temp = self.read_image_input(img)
                    exported_regions, updated_prediction_result = self.craft_detect(temp)
                    inter_text_list, conf_list = self.text_recognize_batch(exported_regions)
                    final_result = self.output_formatter(inter_text_list, conf_list, updated_prediction_result)
                    text_list.append(final_result)

            else:
                image_list = [self.read_image_input(img) for img in image]
                inter_text_list, conf_list = self.text_recognize_batch(image_list)
                final_result = self.output_formatter(inter_text_list, conf_list)
                text_list.extend(final_result)

        # Single image handling
        else:
            image = self.read_image_input(image)

            if self.detect:
                exported_regions, updated_prediction_result = self.craft_detect(image)
                inter_text_list, conf_list = self.text_recognize_batch(exported_regions)
                text_list = [self.output_formatter(inter_text_list, conf_list, updated_prediction_result)]
            else:
                inter_text_list, conf_list = self.text_recognize_batch([image])
                text_list = self.output_formatter(inter_text_list, conf_list)

        return text_list
