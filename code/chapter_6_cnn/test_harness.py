from __future__ import print_function
import copy
import time

# tag::mcts_go_preprocessing[]
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

np.random.seed(123)  # <1>
X = np.load('../generated_games/features-40k.npy')  # <2>
Y = np.load('../generated_games/labels-40k.npy')
W = copy.deepcopy(X)
Z = copy.deepcopy(Y)

samples = X.shape[0]
size = 9
board_size = size ** 2
test_samples = 5000

X = X.reshape(samples, board_size)  # <3>
Y = Y.reshape(samples, board_size)
X_train, X_test = X[:-test_samples], X[-test_samples:]
Y_train, Y_test = Y[:-test_samples], Y[-test_samples:]

W = W.reshape(samples, size, size, 1)
W_train, W_test = W[:-test_samples], W[-test_samples:]
Z_train, Z_test = Z[:-test_samples], Z[-test_samples:]
# end::mcts_go_preprocessing[]

# tag::one_layer_700_mlp_model[]
model_0 = Sequential()
model_0.add(Dense(700, activation='sigmoid', input_shape=(board_size,)))
model_0.add(Dense(board_size, activation='sigmoid'))
model_0.summary()
model_0.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end::one_layer_700_mlp_model[]

# tag::one_layer_2800_mlp_model[]
model_1 = Sequential()
model_1.add(Dense(2800, activation='sigmoid', input_shape=(board_size,)))
model_1.add(Dense(board_size, activation='sigmoid'))
model_1.summary()
model_1.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end::one_layer_2800_mlp_model[]

# tag::three_layer_700_mlp_model[]
model_2 = Sequential()
model_2.add(Dense(200, activation='sigmoid', input_shape=(board_size,)))
model_2.add(Dense(300, activation='sigmoid'))
model_2.add(Dense(200, activation='sigmoid'))
model_2.add(Dense(board_size, activation='sigmoid'))
model_2.summary()
model_2.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_700_mlp_model[]

# tag:: three_layer_1400_mlp_model[]
model_3 = Sequential()
model_3.add(Dense(400, activation='sigmoid', input_shape=(board_size,)))
model_3.add(Dense(600, activation='sigmoid'))
model_3.add(Dense(400, activation='sigmoid'))
model_3.add(Dense(board_size, activation='sigmoid'))
model_3.summary()
model_3.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_1400_mlp_model[]

# tag:: three_layer_224_cnn_model[]
model_4 = Sequential()
model_4.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_4.add(Conv2D(64, (3, 3), activation='sigmoid'))
model_4.add(Flatten())
model_4.add(Dense(128, activation='sigmoid'))
model_4.add(Dense(size**2, activation='sigmoid'))
model_4.summary()
model_4.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_224_cnn_model[]

# tag:: four_layer_224_cnn_model[]
model_5 = Sequential()
model_5.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)))
model_5.add(Dropout(0.6))
model_5.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_5.add(MaxPooling2D((2, 2)))
model_5.add(Dropout(0.6))
model_5.add(Flatten())
model_5.add(Dense(128, activation='relu'))
model_5.add(Dropout(0.6))
model_5.add(Dense(size**2, activation='softmax'))
model_5.summary()
model_5.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: four_layer_224_cnn_model[]

# start_tests
time.clock()
x, y, z = 8, 6, 4
scores = [[[0 for a in range(z)] for b in range(y)] for c in range(x)]
times = [[0 for b in range(y)] for c in range(x)]
model = [model_0, model_1, model_2, model_3, model_4, model_5]
names = ['700n 1d mlp', '2800n 1d mlp', '700n 3d mlp', '1400n 3d mlp', '224n 2c1d cnn', '224n 2c1m1d cnn']

for n in range(x):
    for i in range(y):
        start_time = time.clock()
        if i < 4:
            model[i].fit(X_train, Y_train, batch_size=128, epochs=2**n, verbose=1, validation_data=(X_test, Y_test))
            times[n][i] = time.clock() - start_time;
            scores[n][i][0], scores[n][i][1] = model[i].evaluate(X_train, Y_train, verbose=1)
            scores[n][i][2], scores[n][i][3] = model[i].evaluate(X_test, Y_test, verbose=1)
        else:
            model[i].fit(W_train, Z_train, batch_size=128, epochs=2**n, verbose=1, validation_data=(W_test, Z_test))
            times[n][i] = time.clock() - start_time;
            scores[n][i][0], scores[n][i][1] = model[i].evaluate(W_train, Z_train, verbose=1)
            scores[n][i][2], scores[n][i][3] = model[i].evaluate(W_test, Z_test, verbose=1)

for i in range(y):
    cumulative = 0
    for n in range(x):
        print('Statistics for', names[i], ', after', 2**n, 'epochs.')
        print('Training test loss:', scores[n][i][0] * 100)
        print('Validate test loss:', scores[n][i][2] * 100)
        print('Training test accuracy:', scores[n][i][1] * 100)
        print('Validate test accuracy:', scores[n][i][3] * 100)
        cumulative += times[n][i]
        print('Total training time:', cumulative, '\n');
# end_tests
