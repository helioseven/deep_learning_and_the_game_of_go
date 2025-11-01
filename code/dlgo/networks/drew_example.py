from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D

def layers(input_shape):
    return [
        ZeroPadding2D(padding=1, data_format="channels_last", input_shape=input_shape),
        Conv2D(48, (3, 3), data_format="channels_last"),
        Activation("relu"),

        Conv2D(48, (5, 5), data_format="channels_last"),
        Activation("relu"),

        Conv2D(48, (5, 5), data_format="channels_last"),
        Activation("relu"),

        Conv2D(64, (7, 7), data_format="channels_last"),
        Activation("relu"),
        Dropout(0.1),

        MaxPooling2D((3, 3), strides=(1,1), data_format="channels_last"),
        Dropout(0.3),
        Flatten(),

        Dense(512),
        Activation("relu"),
        Dropout(0.5)
    ]
