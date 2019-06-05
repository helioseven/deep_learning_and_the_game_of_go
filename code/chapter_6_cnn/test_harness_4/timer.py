import time

from keras.callbacks import Callback

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.train_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.train_start)
