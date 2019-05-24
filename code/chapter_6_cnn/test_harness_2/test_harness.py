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
X = np.load('../../generated_games/features-40k.npy')
Y = np.load('../../generated_games/labels-40k.npy')

samples = X.shape[0]
size = 9
test_samples = 5000

X = X.reshape(samples, size, size, 1)

X_train, X_test = X[:-test_samples], X[-test_samples:]
Y_train, Y_test = Y[:-test_samples], Y[-test_samples:]
# end::mcts_go_preprocessing[]

# tag:: two_layer_3x3c_128d_cnn_model[]
model_0 = Sequential()
model_0.add(Conv2D(64, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_0.add(Flatten())
model_0.add(Dense(128, activation='sigmoid'))
model_0.add(Dense(size**2, activation='sigmoid'))
model_0.summary()
model_0.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: two_layer_3x3c_128d_ccn_model[]

# tag:: two_layer_3x3c_256d_cnn_model[]
model_1 = Sequential()
model_1.add(Conv2D(64, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_1.add(Flatten())
model_1.add(Dense(256, activation='sigmoid'))
model_1.add(Dense(size**2, activation='sigmoid'))
model_1.summary()
model_1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: two_layer_3x3c_256d_ccn_model[]

# tag:: two_layer_5x5c_128d_cnn_model[]
model_2 = Sequential()
model_2.add(Conv2D(64, (5, 5), activation='sigmoid', input_shape=(size, size, 1)))
model_2.add(Flatten())
model_2.add(Dense(128, activation='sigmoid'))
model_2.add(Dense(size**2, activation='sigmoid'))
model_2.summary()
model_2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: two_layer_5x5c_128d_ccn_model[]

# tag:: three_layer_3x3c_3x3c_128d_cnn_model[]
model_3 = Sequential()
model_3.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_3.add(Conv2D(64, (3, 3), activation='sigmoid'))
model_3.add(Flatten())
model_3.add(Dense(128, activation='sigmoid'))
model_3.add(Dense(size**2, activation='sigmoid'))
model_3.summary()
model_3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_3x3c_3x3c_128d_cnn_model[]

# tag:: three_layer_3x3c_3x3c_256d_cnn_model[]
model_4 = Sequential()
model_4.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(size, size, 1)))
model_4.add(Conv2D(64, (3, 3), activation='sigmoid'))
model_4.add(Flatten())
model_4.add(Dense(256, activation='sigmoid'))
model_4.add(Dense(size**2, activation='sigmoid'))
model_4.summary()
model_4.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end:: three_layer_3x3c_3x3c_256d_cnn_model[]

# tag:: four_layer_3x3c_3x3c_2x2mp_128d_cnn_model[]
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
# end:: four_layer_3x3c_3x3c_2x2mp_128d_cnn_model[]

# start_tests
m = 6
model = [model_0, model_1, model_2, model_3, model_4, model_5]
colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
names = [
    '1_64-3x3_128d_cnn',
    '1_64-3x3_256d_cnn',
    '1_64-5x5_128d_cnn',
    '2_32x64-3x3_3x3-128d_cnn',
    '2_32x64-3x3_3x3-256d_cnn',
    '3_32x64xMP-3x3_3x3_2x2-128d_cnn']
e = 2**1
callback_times = TimeHistory()
model_histories = [{} for x in range(m)]
model_times = [[] for x in range(m)]

for i in range(m):
    model_histories[i] = model[i].fit(
        X_train,
        Y_train,
        batch_size=64,
        callbacks=[callback_times],
        epochs=e,
        verbose=1,
        validation_data=(X_test, Y_test))
    model_times[i] = callback_times.times
    model[i].save(names[i] + '.h5')

file1 = 'loss_history.png'
file2 = 'test_history.png'

fig = plt.figure()
for i in range(m):
    model_history = [x * 100 for x in model_histories[i].history['loss']]
    plt.loglog(model_times[i], model_history, colors[i])
plt.title('Loss / Time')
plt.ylabel('Loss')
plt.xlabel('Seconds')
names_2 = [x + '_loss' for x in names]
plt.legend(names_2, bbox_to_anchor=(1.04,0.5), loc='center left')
fig.savefig(file1, dpi=250, bbox_inches='tight')

fig = plt.figure()
for i in range(m):
    model_history = [x * 100 for x in model_histories[i].history['acc']]
    plt.plot(callback_times.times, model_history, colors[i] + '-')
    model_history = [x * 100 for x in model_histories[i].history['val_acc']]
    plt.plot(callback_times.times, model_history, colors[i] + '--')
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
fig.savefig(file2, dpi=250, bbox_inches='tight')
# end_tests
