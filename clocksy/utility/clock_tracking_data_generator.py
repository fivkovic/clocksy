import random

import numpy as np
from PIL import Image, ImageDraw

from utility import data_helper, parameters
from utility.data_generator import DataGenerator
from utility.data_helper import ImagePaths, ImageProperties, DataGeneratorType
from utility.log_helper import log_info


class ClockTrackingDataGenerator(DataGenerator):

    def __init__(self, image_paths, image_properties, noise_threshold, data_generator_type):
        super(ClockTrackingDataGenerator, self).__init__(image_paths, image_properties, noise_threshold, data_generator_type)

    def generate_data(self, image_count):

        INPUT_WH = 100 # TODO: Change to be a parameter

        images = np.zeros((image_count, self.image_width, self.image_height, self.image_channels), np.float32)
        labels = np.zeros((image_count, INPUT_WH * 2), np.float32)

        for current_image_id in range(0, image_count):
            random_background_image = self.generate_random_image()

            if random.random() > self.noise_threshold:
                clock_image = self.generate_single_clock_image(data_helper.get_random_angle(),
                                                               data_helper.get_random_angle(),
                                                               data_helper.get_random_angle(),
                                                               transparent = True)

                scaling = data_helper.get_bounding_box_scale(0.25, 0.85, random.random())
                aspect_ratio = 0.85 + random.random() * 0.15

                x_scaling = scaling * aspect_ratio
                y_scaling = scaling / aspect_ratio
                width = int(self.image_width * x_scaling)
                height = int(self.image_height * y_scaling)
                x_start = int(self.image_width * (random.random() - x_scaling / 2))
                y_start = int(self.image_height * (random.random() - y_scaling / 2))
                x_start = 0 if x_start < 0 else x_start
                y_start = 0 if y_start < 0 else y_start
                if x_start + width > self.image_width:
                    x_start = self.image_width - width
                if y_start + height > self.image_height:
                    y_start = self.image_height - height
                x_end = (x_start + width)
                y_end = (y_start + height)

                clock_image = clock_image.resize((width, height), Image.ANTIALIAS)

                rotation_offset = 0
                rotated_clock_image = clock_image.rotate(rotation_offset)

                random_background_image.paste(rotated_clock_image, (x_start, y_start), rotated_clock_image)

                step_x = (self.image_width / INPUT_WH)
                step_y = (self.image_height / INPUT_WH)
                for x in range(0, INPUT_WH):
                    for y in range(0, INPUT_WH):
                        current_x = x * step_x
                        current_y = y * step_y
                        if current_x + step_x >= x_start and current_x <= x_end:
                            labels[current_image_id][x] = 1
                        if current_y + step_y >= y_start and current_y <= y_end:
                            labels[current_image_id][INPUT_WH + y] = 1

            random_background_image = random_background_image.convert('L')

            generated_image = np.array(random_background_image).reshape(self.image_width, self.image_height, self.image_channels)
            np.copyto(images[current_image_id], generated_image)

            images[current_image_id] /= 255.0

        return images, labels


def generate_clock_tracking_images(image_paths, image_properties, count):

    log_info("Initializing clock tracking data generator for clock type %s..." % image_paths.clock_type)
    generator = ClockTrackingDataGenerator(image_paths, image_properties, noise_threshold = 0.2, data_generator_type = DataGeneratorType.HOURS_AND_MINUTES)

    log_info("Images to be generated: %d" % count)
    log_info("Generating clock tracking images...")
    images, labels = generator.generate_data(image_count = count)

    log_info("Processing generated images...")
    for n in range(0, len(images)):
        reshaped_image = images[n].reshape(image_properties.width, image_properties.height) * 255.0
        generated_image = Image.fromarray(reshaped_image).convert("RGB")

        drawing = ImageDraw.Draw(generated_image)
        start_point, end_point = data_helper.calculate_bounding_box(labels[n], image_properties)
        drawing.rectangle((start_point.x, start_point.y, end_point.x, end_point.y), outline = "blue")

        label = data_helper.generate_bounding_box_label(start_point, end_point)
        image_name = "clock_%s_%s_%d.png" % (label, image_paths.clock_type, n)
        generated_image.save(parameters.CLOCK_TRACKING_OUTPUT_PATH + image_name)

    log_info("Images for clock type %s generated successfully." % image_paths.clock_type)

if __name__ == "__main__":

    image_paths_collection = [ImagePaths(clock_type = ct) for ct in parameters.AVAILABLE_CLOCK_TYPES]

    image_properties = ImageProperties(width=parameters.OUTPUT_IMAGE_WIDTH,
                                       height=parameters.OUTPUT_IMAGE_HEIGHT,
                                       channels=parameters.OUTPUT_IMAGE_CHANNELS)

    for image_paths in image_paths_collection:
        generate_clock_tracking_images(image_paths, image_properties, 128)