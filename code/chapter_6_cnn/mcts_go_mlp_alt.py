from __future__ import print_function

# tag::mcts_go_mlp_preprocessing[]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(123)  # <1>
# TODO: tell readers where to put file
X = np.load('../generated_games/features-40k.npy')  # <2>
Y = np.load('../generated_games/labels-40k.npy')
samples = X.shape[0]
board_size = 9 * 9

X = X.reshape(samples, board_size)  # <3>
Y = Y.reshape(samples, board_size)

test_samples = 5000
X_train, X_test = X[:-test_samples], X[-test_samples:]
Y_train, Y_test = Y[:-test_samples], Y[-test_samples:]
# end::mcts_go_mlp_preprocessing[]

# tag::mcts_go_mlp_model[]
model = Sequential()
model.add(Dense(2400, activation='sigmoid', input_shape=(board_size,)))
model.add(Dense(board_size, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_train, Y_train, verbose=0)
print('Training test loss:', score[0] * 100)
print('Training test accuracy:', score[1] * 100)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Validation test loss:', score[0] * 100)
print('Validation test accuracy:', score[1] * 100)
# end::mcts_go_mlp_model[]
