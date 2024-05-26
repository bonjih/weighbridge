import numpy as np
import cv2


class RegionMasking:
    """
    creates the mask for object movement for a specific ROI

    """

    def __init__(self, region):
        self.region = [tuple(pt) for pt in region]

    def isolate(self, img):

        mask = np.zeros_like(img)
        match = (255,) * img.shape[2]
        cv2.fillPoly(mask, [np.array(self.region)], match)
        masked = cv2.bitwise_and(img, mask)
        return masked

    def masking(self, prev_frame, current_frame, back):
        roi1 = self.isolate(prev_frame)
        roi2 = self.isolate(current_frame)
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.pyrDown(gray1)
        frame2 = cv2.pyrDown(gray2)
        blur1 = cv2.GaussianBlur(frame1, (85, 3), 0)
        blur2 = cv2.GaussianBlur(frame2, (85, 3), 0)
        fgmask1 = back.apply(blur1)
        fgmask2 = back.apply(frame2)
        fgmask1[fgmask1 == 200] = 0
        fgmask2[fgmask2 == 200] = 0
        return fgmask1, fgmask2, self.region

    def threshold(self, fgmask1, fgmask2):
        _, thresh1 = cv2.threshold(fgmask1, 0, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(fgmask2, 0, 255, cv2.THRESH_BINARY)
        dilate = cv2.dilate(thresh2, np.ones((1, 1), np.uint8), iterations=1)
        cont_thresh_min = 1000  # if combined with an object tracker - tweak to get optimal threshold
        cont_thresh_max = 2000

        return dilate
