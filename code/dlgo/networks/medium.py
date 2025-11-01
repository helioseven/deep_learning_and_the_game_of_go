from __future__ import absolute_import
from keras.layers import Activation, Conv2D, Dense, Flatten, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, (5, 5), padding='valid', data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]
