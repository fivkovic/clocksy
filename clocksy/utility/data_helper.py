import enum
import random

import numpy as np

from utility import parameters

# ============================================================================================
# CLASSES
# ============================================================================================


class ImagePaths:

    def __init__(self, clock_type):
        self.clock_type = clock_type
        self.clock = parameters.CLOCK_IMAGE_PATH % clock_type
        self.hours_hand = parameters.HOURS_HAND_IMAGE_PATH % clock_type
        self.minutes_hand = parameters.MINUTES_HAND_IMAGE_PATH % clock_type
        self.seconds_hand = parameters.SECONDS_HAND_IMAGE_PATH % clock_type


class ImageProperties:

    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels


class DataGeneratorType(enum.Enum):
    HOURS_ONLY = 1
    HOURS_AND_MINUTES = 2
    HOURS_MINUTES_AND_SECONDS = 3


class Point:

    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y


class Time:

    def __init__(self, is_clock = False, is_clock_confidence = 0.0,
                 hour = 0, hour_confidence = 100.0,
                 minute = 0, minute_confidence = 100.0,
                 second = 0, second_confidence = 100.0):

        self.is_clock = is_clock
        self.is_clock_confidence = is_clock_confidence
        self.hour = hour
        self.hour_confidence = hour_confidence
        self.minute = minute
        self.minute_confidence = minute_confidence
        self.second = second
        self.second_confidence = second_confidence

# ============================================================================================
# FUNCTIONS
# ============================================================================================


def calculate_bounding_box(label_array, image_properties, confidence=0.5):
    IMAGE_WH = int(len(label_array) / 2)
    x_start, y_start, x_end, y_end = 1, 1, 0, 0

    for x in range(0, IMAGE_WH):
        for y in range(0, IMAGE_WH):
            current_x = (x * 1.0 / IMAGE_WH)
            current_y = (y * 1.0 / IMAGE_WH)
            if label_array[x] >= confidence and label_array[IMAGE_WH + y] >= confidence:
                x_start = current_x if current_x < x_start else x_start
                y_start = current_y if current_y < y_start else y_start
                x_end = current_x if current_x > x_end else x_end
                y_end = current_y if current_y > y_end else y_end

    start_point = Point(x_start * image_properties.width, y_start * image_properties.height)
    end_point = Point(x_end * image_properties.width, y_end * image_properties.height)

    return start_point, end_point


def generate_bounding_box_label(start_point, end_point):
    return "%d_%d_%d_%d" % (start_point.x, start_point.y, end_point.x, end_point.y)


def calculate_time(label_array, data_generator_type):
    if label_array[0] > 0.5:
        return Time()

    hour = np.argmax(label_array[1:13])
    hour_confidence = label_array[hour + 1] * 100
    minute = np.argmax(label_array[13:73])
    minute_confidence = label_array[minute + 13] * 100
    if data_generator_type == DataGeneratorType.HOURS_MINUTES_AND_SECONDS:
        second = np.argmax(label_array[73:133])
        second_confidence = label_array[second + 73] * 100
        return Time(True, label_array[0] * 100, hour, hour_confidence, minute, minute_confidence, second, second_confidence)
    else:
        return Time(True, label_array[0] * 100, hour, hour_confidence, minute, minute_confidence)


def generate_time_label(time):
    return "%d_%02d_%02d_%02d" % (time.is_clock, time.hour, time.minute, time.second)


def calculate_is_clock(label_array):
    return label_array[0]


def calculate_position_offset(offset_deviation):
    offset = (int(random.random() * offset_deviation - offset_deviation / 2),
              int(random.random() * offset_deviation - offset_deviation / 2))

    return offset


def calculate_rotation_offset(offset_deviation):
    offset = ((random.random() * offset_deviation - offset_deviation / 2) * 2)

    return offset


def calculate_scaling(scaling_deviation):
    scaling = int(128 + ((random.random() - 0.5) * scaling_deviation * 2.0))
    return scaling, scaling


def get_random_angle():
    return random.random() * 360


def get_bounding_box_scale(start, end, value):
    end -= start
    return start + end * value * value
