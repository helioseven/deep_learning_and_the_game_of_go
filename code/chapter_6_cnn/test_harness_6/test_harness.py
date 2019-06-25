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

# start_keras_models
model_0 = Sequential()
model_0.add(Conv2D(77, (2, 2), activation='relu', input_shape=(size, size, 1)))
model_0.add(Conv2D(128, (2, 2), activation='relu'))
model_0.add(Conv2D(256, (2, 2), activation='relu'))
model_0.add(Dropout(0.1))
model_0.add(MaxPooling2D((2, 2)))
model_0.add(Dropout(0.5))
model_0.add(Flatten())
model_0.add(Dense(256, activation='relu'))
model_0.add(Dropout(0.9))
model_0.add(Dense(size**2, activation='softmax'))
model_0.summary()
model_0.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_1 = Sequential()
model_1.add(Conv2D(77, (2, 2), activation='relu', input_shape=(size, size, 1)))
model_1.add(Conv2D(128, (2, 2), activation='relu'))
model_1.add(Conv2D(256, (2, 2), activation='relu'))
model_1.add(MaxPooling2D((2, 2)))
model_1.add(Dropout(0.5))
model_1.add(Flatten())
model_1.add(Dense(256, activation='relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(size**2, activation='softmax'))
model_1.summary()
model_1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'''
model_2 = Sequential()
model_2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)))
model_2.add(Conv2D(64, (3, 3), activation='relu'))
model_2.add(MaxPooling2D((2, 2), (1, 1)))
model_2.add(Dropout(0.6))
model_2.add(Flatten())
model_2.add(Dense(256, activation='relu'))
model_2.add(Dropout(0.6))
model_2.add(Dense(size**2, activation='softmax'))
model_2.summary()
model_2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_3 = Sequential()
model_3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)))
model_3.add(Conv2D(64, (3, 3), activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(MaxPooling2D((3, 3), (1, 1), 'same'))
model_3.add(Dropout(0.5))
model_3.add(Flatten())
model_3.add(Dense(256, activation='relu'))
model_3.add(Dropout(0.8))
model_3.add(Dense(size**2, activation='softmax'))
model_3.summary()
model_3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_4 = Sequential()
model_4.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)))
model_4.add(Conv2D(64, (3, 3), activation='relu'))
model_4.add(MaxPooling2D((2, 2)))
model_4.add(Dropout(0.6))
model_4.add(Flatten())
model_4.add(Dense(256, activation='relu'))
model_4.add(Dropout(0.6))
model_4.add(Dense(size**2, activation='softmax'))
model_4.summary()
model_4.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_5 = Sequential()
model_5.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 1)))
model_5.add(Dropout(0.6))
model_5.add(Conv2D(64, (3, 3), activation='relu'))
model_5.add(MaxPooling2D((2, 2)))
model_5.add(Dropout(0.6))
model_5.add(Flatten())
model_5.add(Dense(128, activation='relu'))
model_5.add(Dropout(0.6))
model_5.add(Dense(size**2, activation='softmax'))
model_5.summary()
model_5.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
'''
# end_keras_models

# start_tests
m = 2
# model = [model_0, model_1, model_2, model_3, model_4, model_5]
model = [model_0, model_1]
colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
names = [
    '3-conv_2x2_159do',
    '3-conv_2x2_55do',
    'c32-c64-2mp1-do6-d-do6-d',
    'c32-c64-do2-3mp1-do5-d-do8-d',
    'c32-c64-2mp2-do6-d-do6-d',
    'c32-do6-c64-2mp2-do6-d-do6-d']
e = 2**11
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
fig.savefig(file1, dpi=1000, bbox_inches='tight')

fig = plt.figure()
for i in range(m):
    model_history = [x * 100 for x in model_histories[i].history['acc']]
    plt.plot(model_times[i], model_history, colors[i] + '-', linewidth=0.75)
    model_history = [x * 100 for x in model_histories[i].history['val_acc']]
    plt.plot(model_times[i], model_history, colors[i] + '--', linewidth=0.75)
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
fig.savefig(file2, dpi=1000, bbox_inches='tight')
# end_tests
