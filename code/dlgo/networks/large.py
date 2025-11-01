from __future__ import absolute_import
from keras.layers import Activation, Conv2D, Dense, Flatten, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, (7, 7), padding='valid', data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(48, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(48, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(32, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(32, (5, 5), data_format='channels_last'),
        Activation('relu'),

        Flatten(),
        Dense(1024),
        Activation('relu'),
    ]
