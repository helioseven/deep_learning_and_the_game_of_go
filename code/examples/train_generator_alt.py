# tag::train_generator_imports[]
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint  # <1>

# <1> With model checkpoints we can store progress for time-consuming experiments
# end::train_generator_imports[]

# tag::train_generator_generator[]
go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100

encoder = OnePlaneEncoder((go_board_rows, go_board_cols))  # <1>

processor = GoDataProcessor(encoder=encoder.name())  # <2>

generator = processor.load_go_data('train', num_games, use_generator=True)  # <3>
test_generator = processor.load_go_data('test', num_games, use_generator=True)

# <1> First we create an encoder of board size.
# <2> Then we initialize a Go Data processor with it.
# <3> From the processor we create two data generators, for training and testing.
# end::train_generator_generator[]

# tag::train_generator_model[]
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = large.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# end::train_generator_model[]

# tag::train_generator_fit[]
callback_times = TimeHistory()
epochs = 500
batch_size = 128
model_histories = model.fit_generator(generator=generator.generate(batch_size, num_classes),  # <1>
                    epochs=epochs,
                    steps_per_epoch=generator.get_num_samples() / batch_size,  # <2>
                    validation_data=test_generator.generate(batch_size, num_classes),  # <3>
                    validation_steps=test_generator.get_num_samples() / batch_size,  # <4>
                    callbacks=[ModelCheckpoint('../checkpoints/large_model_epoch_{epoch}.h5', period=10), callback_times])  # <5>

model.evaluate_generator(generator=test_generator.generate(batch_size, num_classes),
                         steps=test_generator.get_num_samples() / batch_size)  # <6>
# <1> We specify a training data generator for our batch size...
# <2> ... and how many training steps per epoch we execute.
# <3> An additional generator is used for validation...
# <4> ... which also needs a number of steps.
# <5> After each epoch we persist a checkpoint of the model.
# <6> For evaluation we also speficy a generator and the number of steps.
# end::train_generator_fit[]

model_times = callback_times.times
file1 = '../checkpoints/loss_history.png'
file2 = '../checkpoints/test_history.png'

fig = plt.figure()
model_history = [x * 100 for x in model_histories.history['loss']]
plt.loglog(model_times, model_history)
plt.title('Loss / Time')
plt.ylabel('Loss')
plt.xlabel('Seconds')
plt.legend(['large_nn'], bbox_to_anchor=(1.04,0.5), loc='center left')
fig.savefig(file1, dpi=1000, bbox_inches='tight')

fig = plt.figure()
model_history = [x * 100 for x in model_histories.history['acc']]
plt.plot(model_times, model_history, '-', linewidth=0.75)
model_history = [x * 100 for x in model_histories.history['val_acc']]
plt.plot(model_times[i], model_history, '--', linewidth=0.75)
plt.title('Test % / Time')
plt.ylabel('Test %')
plt.xlabel('Seconds')
plt.legend(['large_nn_train', 'large_nn_valid'], bbox_to_anchor=(1.04,0.5), loc='center left')
fig.savefig(file2, dpi=1000, bbox_inches='tight')