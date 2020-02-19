import argparse

import keras
import numpy as np

from clock_tracking import model
from clock_tracking import parameters
from utility import data_helper
from utility.clock_tracking_data_generator import ClockTrackingDataGenerator
from utility.data_helper import ImagePaths, ImageProperties, DataGeneratorType
from utility.log_helper import log_info


def parse_cli_arguments():
    """
    Parses command line arguments.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(prog='clock_tracking',
                                     description = "Clock tracking module.")
    parser.add_argument("--mode", "-m",
                        type = str,
                        default = parameters.DEFAULT_MODE,
                        help = "Running mode (train, test etc.)")
    parser.add_argument("--model-name", "-mn",
                        type = str,
                        default = parameters.DEFAULT_MODEL_NAME,
                        help = "Name of the trained output model.")
    parser.add_argument("--images-count", "-ic",
                        type = int,
                        default = parameters.DEFAULT_IMAGES_COUNT,
                        help = "Number of total images.")
    parser.add_argument("--batch-size", "-bs",
                        type = int,
                        default = parameters.DEFAULT_BATCH_SIZE,
                        help = "Training batch size.")
    parser.add_argument("--epochs", "-e",
                        type = int,
                        default = parameters.DEFAULT_NUMBER_OF_EPOCHS,
                        help = "Number of epochs.")
    parser.add_argument("--validation-split", "-vs",
                        type = float,
                        default = parameters.DEFAULT_VALIDATION_SPLIT,
                        help = "Validation split [0, 1].")
    parser.add_argument("--noise-threshold", "-nt",
                        type = float,
                        default = parameters.DEFAULT_NOISE_THRESHOLD,
                        help = "Percent of images without clocks [0, 1].")

    return parser.parse_args()

def train(images_count, batch_size, epochs, validation_split, noise_threshold):
    """
    Performs training of the clock tracking model with given parameters.

    :param images_count: Count of images for all clock types.
    :param batch_size: Number of samples per gradient update.
    :param epochs: Number of iterations over the entire x and y data provided for training.
    :param validation_split: Fraction of the training data to be used as validation data.
    :param noise_threshold: Fraction of the generated images containing no clocks.
    """

    log_info("Preparing training...")
    image_paths_collection = [ImagePaths(clock_type = clock_type) for clock_type in parameters.AVAILABLE_CLOCK_TYPES]
    image_properties = ImageProperties(width = parameters.IMAGE_WIDTH, height = parameters.IMAGE_HEIGHT, channels = parameters.IMAGE_CHANNELS)

    images_per_clock_type = int(images_count / len(image_paths_collection))

    log_info("Initializing the model for clock tracking...")
    clock_tracking_model = model.initialize_model()

    training_images, training_labels = None, None
    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = ClockTrackingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=DataGeneratorType.HOURS_AND_MINUTES)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        training_images = current_clock_type_images if training_images is None else np.concatenate((training_images, current_clock_type_images), axis=0)
        training_labels = current_clock_type_labels if training_labels is None else np.concatenate((training_labels, current_clock_type_labels), axis=0)

    log_info("Running %d epochs of training with batch size %d and validation split of %d %%..." % (epochs, batch_size, validation_split * 100))
    clock_tracking_model.fit(training_images, training_labels, batch_size=batch_size, validation_split=validation_split, epochs=epochs, shuffle=True, verbose=2)

    log_info("Finished training. Saving the model...")
    clock_tracking_model.save(parameters.DEFAULT_MODEL_NAME)

    log_info("Model saved under name %s." % arguments.model_name)

def test(images_count, noise_threshold):
    """
    Performs testing of the clock tracking model.

    :param images_count: Count of images for all clock types.
    :param noise_threshold: Fraction of the generated images containing no clocks.
    """

    log_info("Preparing test...")
    image_paths_collection = [ImagePaths(clock_type = clock_type) for clock_type in parameters.AVAILABLE_CLOCK_TYPES]
    image_properties = ImageProperties(width = parameters.IMAGE_WIDTH, height = parameters.IMAGE_HEIGHT, channels = parameters.IMAGE_CHANNELS)

    images_per_clock_type = int(images_count / len(image_paths_collection))

    log_info("Loading the trained model %s for clock tracking..." % arguments.model_name)
    clock_tracking_model = keras.models.load_model(arguments.model_name)

    test_images, test_labels = None, None
    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = ClockTrackingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=DataGeneratorType.HOURS_AND_MINUTES)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        test_images = current_clock_type_images if test_images is None else np.concatenate((test_images, current_clock_type_images), axis=0)
        test_labels = current_clock_type_labels if test_labels is None else np.concatenate((test_labels, current_clock_type_labels), axis=0)

    log_info("Running predictions...")
    predictions = clock_tracking_model.predict(test_images)

    log_info("Calculating...")
    correct, acceptable = 0, 0
    for n in range (0, len(predictions)):
        accurate_start_point, accurate_end_point = data_helper.calculate_bounding_box(test_labels[n], image_properties)
        predicted_start_point, predicted_end_point = data_helper.calculate_bounding_box(predictions[n], image_properties)

        diff_x1 = abs(accurate_start_point.x - predicted_start_point.x) / image_properties.width * 100
        diff_y1 = abs(accurate_start_point.y - predicted_start_point.y) / image_properties.height * 100
        diff_x2 = abs(accurate_end_point.x - predicted_end_point.x) / image_properties.width * 100
        diff_y2 = abs(accurate_end_point.y - predicted_end_point.y) / image_properties.height * 100

        tolerance = 5
        if diff_x1 <= tolerance and diff_x2 <= tolerance and diff_y1 <= tolerance and diff_y2 <= tolerance:
            acceptable += 1
            if diff_x1 == 0 and diff_x2 == 0 and diff_y1 == 0 and diff_y2 == 0:
                correct += 1

        #log_info("SAMPLE %04d | DIFFERENCE: (x1: %1d %%, y1: %1d %%) - (x2: %1d %%, y2: %1d %%) " % (n + 1, diff_x1, diff_y1, diff_x2, diff_y2))

    log_info("Testing finished.")
    log_info("Correct (100%% accurate): %f %%" % (correct / len(predictions) * 100))
    log_info("Acceptable (diff < %d %%): %f" % (tolerance, acceptable / len(predictions) * 100))

if __name__ == '__main__':
    arguments = parse_cli_arguments()
    if arguments.mode == "train":
        train(arguments.images_count, arguments.batch_size, arguments.epochs, arguments.validation_split, arguments.noise_threshold)
    elif arguments.mode == "test":
        test(arguments.images_count, arguments.noise_threshold)
