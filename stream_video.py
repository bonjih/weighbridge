import cv2
import numpy as np
from threading import Thread, Event
import time

from process_frame import FrameProcessor


class VideoStreamer(object):
    """

    """

    def __init__(self, link, cam_name, roi_comp=None, offset=0):
        self.cap = cv2.VideoCapture(link)
        self.cam_name = cam_name
        self.link = link
        self.roi_comp = roi_comp
        self.offset = offset
        self.stop_event = Event()
        self.frame_index = 0  # frame index to track frames

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        if self.offset > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.offset * 1.0e3)

        # Read the first frame outside the loop
        ret, prev_frame = self.cap.read()

        back = cv2.createBackgroundSubtractorMOG2()

        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                break

            ret, frame = self.cap.read()

            if not ret:  # Check if end of the video
                self.stop_event.set()
                break

            if self.roi_comp:
                frame = self.apply_configs(frame, prev_frame, back, self.frame_index)

            # Update the previous frame
            prev_frame = frame

            self.frame = frame
            self.frame_index += 1
            #time.sleep(0.01)

        self.cap.release()

    def apply_configs(self, frame, prev_frame, back, frame_index):
        roi_frame = FrameProcessor(self.roi_comp)

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        frame_mask = roi_frame.process_frame(frame, prev_frame, frame_height, frame_width, ts, back, frame_index)

        return frame_mask

    def __call__(self):
        if not self.stop_event.is_set():
            return np.copy(self.frame)

    def is_stopped(self):
        return self.stop_event.is_set()

    @classmethod
    def stream_helper(cls, src, configs, offset):
        video_streams = []

        if isinstance(src, str):
            video_streams.append(cls(src, 'default_cam', configs, offset))
        elif isinstance(src, dict):
            for cam_name, link in src.items():
                video_stream = cls(link, cam_name, configs, offset)
                video_streams.append(video_stream)
        else:
            raise ValueError("Source (src) must be a string or a dictionary")

        while any(not stream.is_stopped() for stream in video_streams):
            try:
                for video_stream in video_streams:
                    if not video_stream.is_stopped():
                        frame = video_stream()
                        if frame is not None:
                            yield frame, video_stream.cam_name

                if cv2.waitKey(1) == ord('q'):
                    break

            except AttributeError:
                pass

        for video_stream in video_streams:
            video_stream.stop_event.set()
            video_stream.thread.join()
            video_stream.cap.release()

        cv2.destroyAllWindows()
