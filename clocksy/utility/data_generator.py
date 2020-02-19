import os
import random
from abc import ABC, abstractmethod

from PIL import Image, ImageEnhance
from keras.utils import Sequence

from utility import data_helper
from utility import parameters
from utility.data_helper import ImagePaths, ImageProperties, DataGeneratorType


class DataGenerator(ABC, Sequence):

    def __init__(self, image_paths : ImagePaths, image_properties : ImageProperties, noise_threshold : float, data_generator_type : DataGeneratorType):

        self.clock_image = Image.open(image_paths.clock, 'r').convert('RGBA')
        self.hours_hand_image = Image.open(image_paths.hours_hand, 'r').convert('RGBA')
        self.minutes_hand_image = Image.open(image_paths.minutes_hand, 'r').convert('RGBA')
        self.seconds_hand_image = Image.open(image_paths.seconds_hand, 'r').convert('RGBA')

        self.image_width = image_properties.width
        self.image_height = image_properties.height
        self.image_channels = image_properties.channels
        self.noise_threshold = noise_threshold

        self.type = data_generator_type

        self.offset_deviation = 16

        self.random_images = self.__load_random_images()

    def __load_random_images(self):
        RANDOM_IMAGES_COUNT = len([name for name in os.listdir(parameters.RANDOM_IMAGES_COLLECTION_PATH)
                                   if os.path.isfile(os.path.join(parameters.RANDOM_IMAGES_COLLECTION_PATH, name))])
        random_images = []
        for i in range(0, RANDOM_IMAGES_COUNT):
            loaded_image = Image.open(parameters.RANDOM_IMAGE_PATH % i, 'r')\
                                .convert("RGBA")\
                                .resize((self.image_width, self.image_height), Image.ANTIALIAS)

            random_images.append(loaded_image)

        return random_images

    def get_random_image(self, image_id):
        return self.random_images[image_id]

    def generate_random_image(self):
        size = (self.image_width, self.image_height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        random_image = Image.new("RGBA", size, color)

        if random.random() > 0.2:
            image = self.get_random_image(random.randint(0, len(self.random_images) - 1))
            random_image.paste(image, (0, 0), random_image)

            if random.random() > 0.5:
                image = self.get_random_image(random.randint(0, len(self.random_images) - 1))
                random_image.paste(image, (0, 0), random_image)

        return random_image

    def generate_single_clock_image(self, hour_hand_angle, minute_hand_angle, seconds_hand_angle, transparent = False):

        position_offset = data_helper.calculate_position_offset(self.offset_deviation)
        rotation_offset = data_helper.calculate_rotation_offset(self.offset_deviation)
        scaling = data_helper.calculate_scaling(self.offset_deviation)
        transparency = 0 if transparent else 255

        clock_image_rotated = self.clock_image.resize(scaling)\
                                                  .rotate(rotation_offset)
        hours_hand_image_rotated = self.hours_hand_image.resize(scaling)\
                                                  .rotate(90 + rotation_offset - hour_hand_angle)
        minutes_hand_image_rotated = self.minutes_hand_image.resize(scaling)\
                                                       .rotate(90 + rotation_offset - minute_hand_angle)
        seconds_hand_image_rotated = self.seconds_hand_image.resize(scaling)\
                                                    .rotate(90 + rotation_offset - seconds_hand_angle)

        generated_image = Image.new("RGBA", (128, 128), (0, 0, 0, transparency))
        if not transparent:
            random_background_image = self.generate_random_image().resize((128, 128))
            generated_image.paste(random_background_image, (0, 0), random_background_image)
        generated_image.paste(clock_image_rotated, position_offset, clock_image_rotated)
        generated_image.paste(hours_hand_image_rotated, position_offset, hours_hand_image_rotated)
        generated_image.paste(minutes_hand_image_rotated, position_offset, minutes_hand_image_rotated)
        if self.type == DataGeneratorType.HOURS_MINUTES_AND_SECONDS:
            generated_image.paste(seconds_hand_image_rotated, position_offset, seconds_hand_image_rotated)

        if random.random() < 0.2:
            enhancer = ImageEnhance.Brightness(generated_image)
            generated_image = enhancer.enhance(0.5)

        generated_image = generated_image.resize((self.image_width, self.image_height), Image.ANTIALIAS)

        return generated_image

    @abstractmethod
    def generate_data(self, image_count):
        pass