import random

import numpy as np
from PIL import Image

from utility import parameters, data_helper
from utility.data_generator import DataGenerator
from utility.data_helper import ImagePaths, ImageProperties, DataGeneratorType
from utility.log_helper import log_info


class TimeReadingDataGenerator(DataGenerator):

    def __init__(self, image_paths, image_properties, noise_threshold, data_generator_type):
        super(TimeReadingDataGenerator, self).__init__(image_paths, image_properties, noise_threshold, data_generator_type)

    def generate_data(self, image_count):
        start_angle = 0
        end_angle = 360

        output_space_dimensionality = 73 if self.type == DataGeneratorType.HOURS_AND_MINUTES else 133

        images = np.zeros((image_count, self.image_width, self.image_height, self.image_channels), np.float32)
        labels = np.zeros((image_count, output_space_dimensionality), np.float32)

        for current_image_id in range(0, image_count):
            if random.random() > self.noise_threshold:
                total_seconds_passed = int((current_image_id / image_count) * 43200)

                hour_percentage = (total_seconds_passed / 3600) / 12
                minute_percentage = ((total_seconds_passed / 60) % 60) / 60
                second_percentage = (total_seconds_passed % 60) / 60

                hour_hand_angle = start_angle + ((end_angle - start_angle) * hour_percentage)
                minute_hand_angle = start_angle + (end_angle - start_angle) * minute_percentage
                seconds_hand_angle = start_angle + (end_angle - start_angle) * second_percentage

                generated_image = self.generate_single_clock_image(hour_hand_angle, minute_hand_angle, seconds_hand_angle).convert('L')
                generated_image_array = np.array(generated_image).reshape(self.image_width, self.image_height, self.image_channels)

                np.copyto(images[current_image_id], generated_image_array)
                images[current_image_id] /= 255.0

                hour_id = int(hour_percentage * 12) + 1
                minute_id = int(minute_percentage * 60) + 13
                second_id = int(second_percentage * 60) + 73

                labels[current_image_id][hour_id] = 1
                labels[current_image_id][minute_id] = 1
                if self.type == DataGeneratorType.HOURS_MINUTES_AND_SECONDS:
                    labels[current_image_id][second_id] = 1
            else:
                random_image = self.generate_random_image().convert('L')
                random_image_array = np.array(random_image).reshape(self.image_width, self.image_height, self.image_channels)
                np.copyto(images[current_image_id], random_image_array)

                images[current_image_id] /= 255.0
                labels[current_image_id][0] = 1

        return images, labels


def generate_time_reading_images(image_paths, image_properties, count):

    log_info("Initializing time reading data generator for clock type %s..." % image_paths.clock_type)
    generator = TimeReadingDataGenerator(image_paths, image_properties, noise_threshold = 0.2, data_generator_type = DataGeneratorType.HOURS_AND_MINUTES)

    log_info("Images to be generated: %d" % count)
    log_info("Generating time reading images...")
    images, labels = generator.generate_data(image_count = count)

    log_info("Processing generated images...")

    for n in range(0, len(images)):
        reshaped_image = images[n].reshape(image_properties.width, image_properties.height) * 255.0
        generated_image = Image.fromarray(reshaped_image).convert("RGB")

        time = data_helper.calculate_time(labels[n], generator.type)

        label = data_helper.generate_time_label(time)
        image_name = "clock_%s_%s_%d.png" % (label, image_paths.clock_type, n)
        generated_image.save(parameters.TIME_READING_OUTPUT_PATH + image_name)

    log_info("Images for clock type %s generated successfully." % image_paths.clock_type)

if __name__ == "__main__":

    image_paths_collection = [ImagePaths(clock_type = ct) for ct in parameters.AVAILABLE_CLOCK_TYPES]

    image_properties = ImageProperties(width=parameters.OUTPUT_IMAGE_WIDTH,
                                       height=parameters.OUTPUT_IMAGE_HEIGHT,
                                       channels=parameters.OUTPUT_IMAGE_CHANNELS)

    for image_paths in image_paths_collection:
        generate_time_reading_images(image_paths, image_properties, 24)