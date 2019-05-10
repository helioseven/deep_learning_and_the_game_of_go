from __future__ import print_function
import matplotlib.pyplot as plt
from timer import TimeHistory
import copy
import math

# tag::mcts_go_preprocessing[]
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

np.random.seed(123)
X = np.load('../generated_games/features-40k.npy')
Y = np.load('../generated_games/labels-40k.npy')
W = copy.deepcopy(X)
Z = copy.deepcopy(Y)

samples = X.shape[0]
size = 9
board_size = size ** 2
test_samples = 5000

X = X.reshape(samples, board_size)
Y = Y.reshape(samples, board_size)
X_train, X_test = X[:-test_samples], X[-test_samples:]
Y_train, Y_test = Y[:-test_samples], Y[-test_samples:]

W = W.reshape(samples, size, size, 1)
W_train, W_test = W[:-test_samples], W[-test_samples:]
Z_train, Z_test = Z[:-test_samples], Z[-test_samples:]
# end::mcts_go_preprocessing[]

# tag:: one_layer_81_mlp_model[]
model_0 = Sequential()
model_0.add(Dense(81, activation='sigmoid', input_shape=(board_size,)))
model_0.add(Dense(board_size, activation='sigmoid'))
model_0.summary()
model_0.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: one_layer_81_mlp_model[]

# tag::one_layer_300_mlp_model[]
model_1 = Sequential()
model_1.add(Dense(300, activation='sigmoid', input_shape=(board_size,)))
model_1.add(Dense(board_size, activation='sigmoid'))
model_1.summary()
model_1.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end::one_layer_300_mlp_model[]

# tag::one_layer_700_mlp_model[]
model_2 = Sequential()
model_2.add(Dense(700, activation='sigmoid', input_shape=(board_size,)))
model_2.add(Dense(board_size, activation='sigmoid'))
model_2.summary()
model_2.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end::one_layer_700_mlp_model[]

# tag:: one_layer_2100_mlp_model[]
model_3 = Sequential()
model_3.add(Dense(2100, activation='sigmoid', input_shape=(board_size,)))
model_3.add(Dense(board_size, activation='sigmoid'))
model_3.summary()
model_3.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: one_layer_2100_mlp_model[]

# tag::three_layer_700_mlp_model[]
model_4 = Sequential()
model_4.add(Dense(200, activation='sigmoid', input_shape=(board_size,)))
model_4.add(Dense(300, activation='sigmoid'))
model_4.add(Dense(200, activation='sigmoid'))
model_4.add(Dense(board_size, activation='sigmoid'))
model_4.summary()
model_4.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_700_mlp_model[]

# tag:: three_layer_224_cnn_model[]
model_5 = Sequential()
model_5.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_5.add(Conv2D(64, (3, 3), activation='sigmoid'))
model_5.add(Flatten())
model_5.add(Dense(128, activation='sigmoid'))
model_5.add(Dense(size**2, activation='sigmoid'))
model_5.summary()
model_5.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_224_cnn_model[]

# tag:: four_layer_224_cnn_model[]
model_6 = Sequential()
model_6.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size, size, 1)))
model_6.add(Dropout(0.6))
model_6.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_6.add(MaxPooling2D((2, 2)))
model_6.add(Dropout(0.6))
model_6.add(Flatten())
model_6.add(Dense(128, activation='relu'))
model_6.add(Dropout(0.6))
model_6.add(Dense(size**2, activation='softmax'))
model_6.summary()
model_6.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# end:: four_layer_224_cnn_model[]

# start_tests
m = 7
model = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]
names = [
    '81n/1_mlp',
    '300n/1_mlp',
    '700n/1_mlp',
    '2100n/1_mlp',
    '700n/3_mlp',
    '224n/2_cnn',
    '224n/3_cnn']
e = 1000
callback_times = TimeHistory()
model_histories = [{} for x in range(m)]
model_times = [[] for x in range(m)]

for i in range(m):
    if i < 5:
        model_histories[i] = model[i].fit(
            X_train,
            Y_train,
            batch_size=64,
            callbacks=[callback_times],
            epochs=e,
            verbose=1,
            validation_data=(X_test, Y_test))
    else:
        model_histories[i] = model[i].fit(
            W_train,
            Z_train,
            batch_size=64,
            callbacks=[callback_times],
            epochs=e,
            verbose=1,
            validation_data=(W_test, Z_test))
    model_times[i] = callback_times.times
    model[i].save(names[i] + '.h5')

file1 = 'loss_history.png'
file2 = 'test_history.png'

fig = plt.figure()
for i in range(m):
    plt.loglog(model_times[i], model_histories[i].history['loss'])
plt.title('MSE Loss / Time')
plt.ylabel('MSE Loss')
plt.xlabel('Seconds')
names_2 = [x + '_loss' for x in names]
plt.legend(names_2, bbox_to_anchor=(1.04,0.5), loc='center left')
fig.savefig(file1, dpi=fig.dpi, bbox_inches='tight')

fig = plt.figure()
for i in range(m):
    plt.semilogx(callback_times.times, model_histories[i].history['acc'])
    plt.semilogx(callback_times.times, model_histories[i].history['val_acc'])
plt.title('Test % / Time')
plt.ylabel('Test %')
plt.xlabel('Seconds')
names_2 = ['' for x in range(len(names) * 2)]
for n in range(len(names_2)):
    if n % 2 == 0:
        names_2[n] = names[math.floor(n/2)] + '_train_acc'
    else:
        names_2[n] = names[math.floor((n-1)/2)] + '_valid_acc'
plt.legend(names_2, bbox_to_anchor=(1.04,0.5), loc='center left')
fig.savefig(file2, dpi=fig.dpi, bbox_inches='tight')
# end_tests
