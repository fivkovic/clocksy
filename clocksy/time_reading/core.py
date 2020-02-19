import argparse

import keras
import numpy as np

from time_reading import model
from time_reading import parameters
from utility import data_helper
from utility.data_helper import ImagePaths, ImageProperties, DataGeneratorType
from utility.log_helper import log_info
from utility.time_reading_data_generator import TimeReadingDataGenerator


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

# TODO: Remove train_old_model
def train_old_model(images_count, batch_size, epochs, validation_split, noise_threshold):
    """
    Performs training of the time reading model with given parameters.

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

    log_info("Initializing the model for time reading...")
    data_generator_type = DataGeneratorType.HOURS_AND_MINUTES
    time_reading_model = model.initialize_old_model(data_generator_type)

    training_images, training_labels = None, None
    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = TimeReadingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=data_generator_type)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        training_images = current_clock_type_images if training_images is None else np.concatenate((training_images, current_clock_type_images), axis=0)
        training_labels = current_clock_type_labels if training_labels is None else np.concatenate((training_labels, current_clock_type_labels), axis=0)

    log_info("Running %d epochs of training with batch size %d and validation split of %d %%..." % (epochs, batch_size, validation_split * 100))
    time_reading_model.fit(training_images, training_labels, batch_size=batch_size, validation_split=validation_split, epochs=epochs, shuffle=True, verbose=2)

    log_info("Finished training. Saving the model...")
    time_reading_model.save(parameters.DEFAULT_MODEL_NAME)

    log_info("Model saved under name %s" % arguments.model_name)

# TODO: Remove test_old_model
def test_old_model(images_count, noise_threshold):
    """
    Performs testing of the time reading model.

    :param images_count: Count of images for all clock types.
    :param noise_threshold: Fraction of the generated images containing no clocks.
    """

    log_info("Preparing test...")
    image_paths_collection = [ImagePaths(clock_type = clock_type) for clock_type in parameters.AVAILABLE_CLOCK_TYPES]
    image_properties = ImageProperties(width = parameters.IMAGE_WIDTH, height = parameters.IMAGE_HEIGHT, channels = parameters.IMAGE_CHANNELS)

    images_per_clock_type = int(images_count / len(image_paths_collection))

    log_info("Loading the trained model %s for time reading..." % arguments.model_name)
    time_reading_model = keras.models.load_model(arguments.model_name)

    test_images, test_labels = None, None
    data_generator_type = DataGeneratorType.HOURS_AND_MINUTES

    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = TimeReadingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=data_generator_type)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        test_images = current_clock_type_images if test_images is None else np.concatenate((test_images, current_clock_type_images), axis=0)
        test_labels = current_clock_type_labels if test_labels is None else np.concatenate((test_labels, current_clock_type_labels), axis=0)

    log_info("Running predictions...")
    predictions = time_reading_model.predict(test_images)

    log_info("Calculating...")
    correct, correct_hours, correct_minutes, acceptable = 0, 0, 0, 0
    for n in range (0, len(predictions)):
        accurate_time = data_helper.calculate_time(test_labels[n], data_generator_type)
        predicted_time = data_helper.calculate_time(predictions[n], data_generator_type)

        log_info("SAMPLE %04d | ACCURATE: %02d:%02d:%02d | PREDICTED: %02d:%02d:%02d" %
                 (n + 1, accurate_time.hour, accurate_time.minute, accurate_time.second,
                  predicted_time.hour, predicted_time.minute, predicted_time.second))

        #is_clock_accurate = int(is_clock_labels[n])
        #is_clock_predicted = np.argmax(predictions[0][n])

        log_info("SAMPLE %04d | ACCURATE: %02d:%02d IC = %d | PREDICTED: %02d:%02d IC = %d" %
                 (n + 1, accurate_time.hour, accurate_time.minute, accurate_time.is_clock, predicted_time.hour, predicted_time.minute,
                  predicted_time.is_clock))

        diff_h = abs(accurate_time.hour - predicted_time.hour)
        diff_m = abs(accurate_time.minute - predicted_time.minute)
        # diff_s = abs(accurate_time.second - predicted_time.second)

        if diff_h == 0:
            correct_hours += 1
        if diff_m == 0:
            correct_minutes += 1

        tolerance = 5
        if accurate_time.is_clock == predicted_time.is_clock and diff_h == 0 and diff_m <= tolerance:
            acceptable += 1
            if diff_h == 0 and diff_m == 0:
                correct += 1

        # log_info("SAMPLE %04d | LOSS: H - %2d | M - %2d" % (n + 1, diff_h, diff_m))
        # log_info("IS CLOCK ACCURATE: %d" % (is_clock_accurate == is_clock_predicted))

    log_info("Testing finished.")
    log_info("Correct HOURS (100%% accurate): %f %%" % (correct_hours / len(predictions) * 100))
    log_info("Correct MINUTES (100%% accurate): %f %%" % (correct_minutes / len(predictions) * 100))
    log_info("Correct (100%% accurate): %f %%" % (correct / len(predictions) * 100))
    log_info("Acceptable (diff < %d minutes): %f" % (tolerance, acceptable / len(predictions) * 100))

    log_info("Testing finished.")

def train(images_count, batch_size, epochs, validation_split, noise_threshold):
    """
    Performs training of the time reading model with given parameters.

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

    log_info("Initializing the model for time reading...")
    data_generator_type = DataGeneratorType.HOURS_AND_MINUTES
    time_reading_model = model.initialize_model(data_generator_type)

    training_images, training_labels = None, None
    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = TimeReadingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=data_generator_type)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        training_images = current_clock_type_images if training_images is None else np.concatenate((training_images, current_clock_type_images), axis=0)
        training_labels = current_clock_type_labels if training_labels is None else np.concatenate((training_labels, current_clock_type_labels), axis=0)

    log_info("Transforming labels...")
    # TODO: Edit the TimeReadingData generator to return appropriate labels?
    hour_labels = [data_helper.calculate_time(training_labels[x], data_generator_type).hour for x in range(0, len(training_labels))]
    minute_labels = [data_helper.calculate_time(training_labels[x], data_generator_type).minute / 60.0 for x in range(0, len(training_labels))]
    is_clock_labels = []
    for x in range(0, len(training_labels)):
        is_clock_labels.append([0, 1]) if training_labels[x][0] else is_clock_labels.append([1, 0])


    log_info("Running %d epochs of training with batch size %d and validation split of %d %%..." % (epochs, batch_size, validation_split * 100))
    time_reading_model.fit(training_images, [is_clock_labels, hour_labels, minute_labels], batch_size=batch_size, validation_split=validation_split, epochs=epochs, shuffle=True, verbose=2)

    log_info("Finished training. Saving the model...")
    time_reading_model.save(parameters.DEFAULT_MODEL_NAME)

    log_info("Model saved under name %s" % arguments.model_name)

def test(images_count, noise_threshold):
    """
    Performs testing of the time reading model.

    :param images_count: Count of images for all clock types.
    :param noise_threshold: Fraction of the generated images containing no clocks.
    """

    log_info("Preparing test...")
    image_paths_collection = [ImagePaths(clock_type = clock_type) for clock_type in parameters.AVAILABLE_CLOCK_TYPES]
    image_properties = ImageProperties(width = parameters.IMAGE_WIDTH, height = parameters.IMAGE_HEIGHT, channels = parameters.IMAGE_CHANNELS)

    images_per_clock_type = int(images_count / len(image_paths_collection))

    log_info("Loading the trained model %s for time reading..." % arguments.model_name)
    dependencies = { 'custom_accuracy': model.custom_accuracy }
    time_reading_model = keras.models.load_model(arguments.model_name, custom_objects=dependencies)

    test_images, test_labels = None, None
    data_generator_type = DataGeneratorType.HOURS_AND_MINUTES

    for image_paths in image_paths_collection:

        log_info("Initializing the clock generator for clock type %s..." % image_paths.clock_type)
        generator = TimeReadingDataGenerator(image_paths, image_properties, noise_threshold=noise_threshold, data_generator_type=data_generator_type)

        log_info("Generating %d images for clock type %s..." % (images_per_clock_type, image_paths.clock_type))
        current_clock_type_images, current_clock_type_labels = generator.generate_data(image_count=images_per_clock_type)

        test_images = current_clock_type_images if test_images is None else np.concatenate((test_images, current_clock_type_images), axis=0)
        test_labels = current_clock_type_labels if test_labels is None else np.concatenate((test_labels, current_clock_type_labels), axis=0)

    log_info("Running predictions...")
    predictions = time_reading_model.predict(test_images)

    log_info("Transforming labels...")
    hour_labels = [data_helper.calculate_time(test_labels[x], data_generator_type).hour for x in range(0, len(test_labels))]
    minute_labels = [data_helper.calculate_time(test_labels[x], data_generator_type).minute / 60.0 for x in range(0, len(test_labels))]
    #is_clock_labels = [test_labels[x][0] for x in range(0, len(test_labels))]
    is_clock_labels = [data_helper.calculate_is_clock(test_labels[x]) for x in range(0, len(test_labels))]

    log_info("Calculating...")
    correct, correct_hours, correct_minutes, in_tolerance_minutes, correct_is_clock, acceptable = 0, 0, 0, 0, 0, 0
    for n in range (0, len(predictions[0])):
        accurate_hour = hour_labels[n]
        predicted_hour = np.argmax(predictions[1][n])

        accurate_minute = (minute_labels[n] * 60)
        predicted_minute = int(predictions[2][n][0] * 60)

        is_clock_accurate = int(is_clock_labels[n])
        is_clock_predicted = np.argmax(predictions[0][n])

        log_info("SAMPLE %04d | ACCURATE: %02d:%02d IC = %d | PREDICTED: %02d:%02d IC = %d" %
                 (n + 1, accurate_hour, accurate_minute, is_clock_accurate, predicted_hour, predicted_minute, is_clock_predicted))

        diff_h = abs(accurate_hour - predicted_hour)
        diff_m = abs(accurate_minute - predicted_minute)

        if diff_h == 0:
            correct_hours += 1
        if diff_m == 0:
            correct_minutes += 1

        tolerance = 5
        if is_clock_accurate == is_clock_predicted and diff_h == 0 and diff_m <= tolerance:
            acceptable += 1
            if diff_h == 0 and diff_m == 0:
                correct += 1

        if diff_m <= tolerance:
            in_tolerance_minutes += 1

        if is_clock_accurate == is_clock_predicted:
            correct_is_clock += 1

        #log_info("SAMPLE %04d | LOSS: H - %2d | M - %2d" % (n + 1, diff_h, diff_m))
        #log_info("IS CLOCK ACCURATE: %d" % (is_clock_accurate == is_clock_predicted))

    log_info("Testing finished.")
    log_info("Correct IS CLOCK (100%% accurate): %f %%" % (correct_is_clock / len(predictions[0]) * 100))
    log_info("Correct HOURS (100%% accurate): %f %%" % (correct_hours / len(predictions[0]) * 100))
    log_info("Correct MINUTES (100%% accurate): %f %%" % (correct_minutes / len(predictions[0]) * 100))
    log_info("In tolerance MINUTES (diff < %d minutes): %f %%" % (tolerance, in_tolerance_minutes / len(predictions[0]) * 100))
    log_info("Correct (100%% accurate): %f %%" % (correct / len(predictions[0]) * 100))
    log_info("Acceptable (diff < %d minutes): %f" % (tolerance, acceptable / len(predictions[0]) * 100))

if __name__ == '__main__':
    arguments = parse_cli_arguments()
    if arguments.mode == "train":
        train(arguments.images_count, arguments.batch_size, arguments.epochs, arguments.validation_split, arguments.noise_threshold)
    elif arguments.mode == "test":
        test(arguments.images_count, arguments.noise_threshold)
    elif arguments.mode == "train_old":
        train_old_model(arguments.images_count, arguments.batch_size, arguments.epochs, arguments.validation_split, arguments.noise_threshold)
    elif arguments.mode == "test_old":
        test_old_model(arguments.images_count, arguments.noise_threshold)