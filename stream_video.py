import cv2
import numpy as np
from threading import Thread, Event
import time

from process_frame import FrameProcessor
import global_params_variables

params = global_params_variables.ParamsDict()
output_video_path = params.get_value('output_video_path')


class VideoStreamer(object):
    """
    A class to stream video from a given source, process the frames, and optionally save the video.
    """

    def __init__(self, link, cam_name, roi_comp=None, offset=0, is_save_video=False):
        self.cap = cv2.VideoCapture(link)
        self.cam_name = cam_name
        self.link = link
        self.roi_comp = roi_comp
        self.offset = offset
        self.is_save_video = is_save_video
        self.stop_event = Event()
        self.frame_index = 0  # frame index to track frames
        self.recording = False
        self.writer = None

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

            if self.is_save_video:
                if self.writer is None:
                    frame_width = int(self.cap.get(3))
                    frame_height = int(self.cap.get(4))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
                    self.output_path = f"{output_video_path}/{self.cam_name}.mp4"
                    self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
                self.writer.write(frame)
            #time.sleep(0.01)

        if self.writer is not None:
            self.writer.release()
        self.cap.release()

    def apply_configs(self, frame, prev_frame, back, frame_index):
        roi_frame = FrameProcessor(self.roi_comp)

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        frame_mask = roi_frame.process_frame(frame, prev_frame, frame_height, frame_width, ts, back, frame_index)

        return frame_mask

    def start_recording(self, output_path):
        self.recording = True
        self.output_path = output_path

    def stop_recording(self):
        self.recording = False

    def __call__(self):
        if not self.stop_event.is_set():
            return np.copy(self.frame)

    def is_stopped(self):
        return self.stop_event.is_set()

    @classmethod
    def stream_helper(cls, src, configs, offset, is_save_video=False, output_path=None):
        video_streams = []

        if isinstance(src, str):
            video_streams.append(cls(src, 'default_cam', configs, offset, is_save_video))
        elif isinstance(src, dict):
            for cam_name, link in src.items():
                video_stream = cls(link, cam_name, configs, offset, is_save_video)
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
