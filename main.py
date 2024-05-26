import cv2

from RoiMultiClass import ComposeROI
import global_params_variables
from process_frame import func_call
from stream_video import VideoStreamer
from utils import clear_directory

params = global_params_variables.ParamsDict()
configs = ComposeROI(params.get_all_items())
video_path = configs.video_file
offset = params.get_value('offset')  # start video at a ts
is_save_video = params.get_value('is_save_video')
output_roi_path = params.get_value('output_roi_path')


def run():
    streamer = VideoStreamer.stream_helper(video_path, configs, offset, is_save_video)

    for frame, cam_name in streamer:
        cv2.imshow(cam_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break


def main():
    print('Processing Video')
    run()
    print('Saving / Segmenting images')
    func_call()

    clear_directory(output_roi_path)


if __name__ == "__main__":
    main()
