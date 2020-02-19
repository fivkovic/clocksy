from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

from clock_tracking.parameters import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, OUTPUT_SPACE_DIMENSIONALITY


def initialize_model():

    model = Sequential()

    model.add(Conv2D(16, (5, 5), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(OUTPUT_SPACE_DIMENSIONALITY))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    return model
