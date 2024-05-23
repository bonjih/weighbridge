import cv2

from RoiMultiClass import ComposeROI
import global_params_variables
from utils import print_results, format_results, save_results
from hf_mnist import predict_numbers
from process_truck import process_images_dir, group_predictions_sort
from stream_video import VideoStreamer

params = global_params_variables.ParamsDict()
configs = ComposeROI(params.get_all_items())
video_path = configs.video_file
offset = params.get_value('offset')  # start video at a ts


def run():
    streamer = VideoStreamer.stream_helper(video_path, configs, offset)

    for frame, cam_name in streamer:
        cv2.imshow(cam_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break


def main():
    print('Processing Video')
    run()
    print('Saving / Segmenting images')
    ROIs = process_images_dir("saved_frames")
    print('Running Prediction')
    predictions = predict_numbers(ROIs)
    predictions = group_predictions_sort(predictions)

    # results of prediction
    results = format_results(predictions)
    print_results(results)
    save_results(results[0], video_path)


if __name__ == "__main__":
    main()
