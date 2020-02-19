import keras
from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential

from utility.data_helper import DataGeneratorType

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_CHANNELS = 1


def initialize_old_model(data_generator_type):

    # TODO: See what to do with the old model

    output_space_dimensionality = 73 if data_generator_type == DataGeneratorType.HOURS_AND_MINUTES else 133

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(output_space_dimensionality))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    return model

# =====================================================================================================================


def custom_accuracy(y_true, y_predicted):
    return K.mean(K.equal(K.round(y_true), K.round(y_predicted)))


def initialize_model(data_generator_type):

    inp = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    x = Conv2D(32, kernel_size=5, strides=2, activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Dropout(0.05)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=3, strides=1, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    isclock = Dense(16, activation='relu')(x)
    isclock = Dense(16, activation='relu')(isclock)
    isclock = Dense(2, activation='softmax', name='isclock')(isclock)

    hour = Dense(288, activation='relu')(x)
    hour = Dense(288, activation='relu')(hour)
    hour = Dense(12, activation='softmax', name='hour')(hour)

    minute = Dense(720, activation='relu')(x)
    minute = Dense(1440, activation='relu')(minute)
    minute = Dense(1, activation='linear', name='minute')(minute)
    # minute = Dense(60, activation='softmax', name='minute')(minute)

    model = Model(inputs=inp, outputs=[isclock, hour, minute])

    model.summary()

    adam_optimizer = keras.optimizers.adam(lr=.001)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'mse'], optimizer=adam_optimizer, metrics=[custom_accuracy, 'accuracy'])

    return model
