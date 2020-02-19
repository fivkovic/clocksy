# ============================================================================================
# PARAMETERS and CONSTANTS
# ============================================================================================

RESOURCES_PATH = "../resources/"

CLOCK_IMAGE_PATH = RESOURCES_PATH + "static/clock_%s.png"
HOURS_HAND_IMAGE_PATH = RESOURCES_PATH + "static/hours_hand_%s.png"
MINUTES_HAND_IMAGE_PATH = RESOURCES_PATH + "static/minutes_hand_%s.png"
SECONDS_HAND_IMAGE_PATH = RESOURCES_PATH + "static/seconds_hand_%s.png"

RANDOM_IMAGES_COLLECTION_PATH = RESOURCES_PATH + "random/"
RANDOM_IMAGE_PATH = RANDOM_IMAGES_COLLECTION_PATH + "%d.png"

AVAILABLE_CLOCK_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

OUTPUT_IMAGE_WIDTH = 100
OUTPUT_IMAGE_HEIGHT = 100
OUTPUT_IMAGE_CHANNELS = 1

RANDOM_DATA_THRESHOLD = 0.2

CLOCK_TRACKING_OUTPUT_PATH = RESOURCES_PATH + "generated/clock_tracking/"
TIME_READING_OUTPUT_PATH = RESOURCES_PATH + "generated/time_reading/"
