import argparse
import numpy as np

import cv2
import keras
from PIL import Image, ImageDraw

from time_reading import model
from utility import data_helper
from utility.data_helper import ImageProperties, DataGeneratorType
from utility.log_helper import log_info


def parse_cli_arguments():
    """
    Parses command line arguments.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(prog='controller',
                                     description = "Controller module.")
    parser.add_argument("--source", "-s",
                        type = str,
                        default = "webcam",
                        help = "Running mode (webcam, stream etc.)")
    parser.add_argument("--stream-url", "-su",
                        type = str,
                        default = "http://192.168.1.102:8080/video",
                        help = "Stream URL")
    parser.add_argument("--width",
                        type = int,
                        default = 640,
                        help = "Video width")
    parser.add_argument("--height",
                        type = int,
                        default = 480,
                        help = "Video height")

    return parser.parse_args()


if __name__ == '__main__':

    log_info("Parsing command line arguments...")
    source = 0
    arguments = parse_cli_arguments()
    if arguments.source == "webcam":
        source = 0
    elif arguments.source == "stream":
        source = arguments.stream_url

    width = arguments.width
    height = arguments.height

    x_scaling = width / 100
    y_scaling = height / 100

    dependencies = {
        'custom_accuracy': model.custom_accuracy
    }

    log_info("Loading clock tracking model...")
    clock_tracking_model = keras.models.load_model('../clock_tracking/clock_tracking.h5')
    log_info("Loading time reading model...")
    time_reading_model = keras.models.load_model('../time_reading/time_reading.h5', custom_objects=dependencies)

    log_info("Initializing video capture...")
    video_capture = cv2.VideoCapture(source)
    video_capture.set(3, width)  # set the resolution
    video_capture.set(4, height)
    (W, H) = (None, None)

    while True:
        (is_grabbed, frame) = video_capture.read()
        if not is_grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        video_output = frame.copy()

        frame = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)  # .astype("float32")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_pil = Image.fromarray(frame)
        array_for_prediction = np.zeros((1, 100, 100, 1), np.float32)
        reshaped_array = np.array(im_pil).reshape(100, 100, 1) / 255.0
        np.copyto(array_for_prediction[0], reshaped_array)

        prediction = clock_tracking_model.predict(array_for_prediction)
        start_point, end_point = data_helper.calculate_bounding_box(prediction[0], ImageProperties(100, 100, 1))

        drawing = ImageDraw.Draw(im_pil)
        drawing.rectangle((start_point.x, start_point.y, end_point.x, end_point.y), outline="green")
        drawing_to_show = np.array(im_pil).reshape(100, 100, 1)
        #cv2.imshow("100x100", drawing_to_show)

        x1 = int(start_point.x * x_scaling)
        y1 = int(start_point.y * y_scaling)
        x2 = int(end_point.x * x_scaling)
        y2 = int(end_point.y * y_scaling)

        video_output = cv2.rectangle(video_output, (x1, y1), (x2, y2), (255, 0, 0), 1)

        video_output_copy = video_output.copy()
        roi = video_output_copy
        if y2 > y1 and x2 > x1 and abs(x2 - x1) > 6 * x_scaling and abs(y2 - y1) > 6 * y_scaling:
            roi = video_output_copy[y1:y2, x1:x2]

        cv2.imshow("ROI", roi)

        frame = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA) #.astype("float32")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_pil = Image.fromarray(frame)
        array_for_prediction = np.zeros((1, 100, 100, 1), np.float32)
        reshaped_array = np.array(im_pil).reshape(100, 100, 1) / 255.0
        np.copyto(array_for_prediction[0], reshaped_array)

        time_prediction = time_reading_model.predict(array_for_prediction)

        hours_to_show = np.argmax(time_prediction[1])
        hours_confidence = time_prediction[1][0][hours_to_show]
        minutes_to_show = int(time_prediction[2][0][0] * 60)
        is_clock = True if np.argmax(time_prediction[0]) == 0 else False

        #predicted_time = data_helper.calculate_time(time_prediction[0], DataGeneratorType.HOURS_AND_MINUTES)

        #predicted_time_text = "IS CLOCK: %d %3d | TIME: %02d:%02d | CONFIDENCE: H - %3d %% M - %3d %%" % \
        #                      (predicted_time.is_clock, predicted_time.is_clock_confidence,
        #                      predicted_time.hour, predicted_time.minute,
        #                      predicted_time.hour_confidence, predicted_time.minute_confidence)

        predicted_time_text = "TIME: %02d:%02d - HCONF = %.2f %%" % (hours_to_show, minutes_to_show, hours_confidence * 100) if is_clock else "No clock found"

        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 690)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.rectangle(video_output, (10, 700), (400, 670), (0, 0, 0), -1)
        cv2.putText(video_output, predicted_time_text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow("Video output", video_output)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    log_info("Disposing...")
    cv2.destroyAllWindows()
    video_capture.release()