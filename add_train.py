from datetime import datetime
import os
import tensorflow as tf

from star_rnn import STARCell
from prepare_data import load_data

BATCH_SIZE = 200
EPOCHS = 30

RNN_UNITS = 32
NUM_LAYERS = 2
DROPOUT = 0.1

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data('add', seq_len=200)

with tf.device('cpu'):
    x_train = tf.constant(x_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    print("X train: ", x_train.shape, "Y train: ", y_train.shape)

    x_valid = tf.constant(x_valid, dtype=tf.float32)
    y_valid = tf.constant(y_valid, dtype=tf.float32)
    print("X val: ", x_valid.shape, "Y val: ", y_valid.shape)

    x_test = tf.constant(x_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)
    print("X test: ", x_test.shape, "Y test: ", y_test.shape)
    print()

time_dim = x_train.shape[1]
channel_dim = x_train.shape[-1]
num_classes = y_train.shape[-1]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

del x_train, y_train, x_valid, y_valid, x_test, y_test

# Model definition
ip = tf.keras.layers.Input(shape=(time_dim, channel_dim))

x = ip
states = None

for i in range(NUM_LAYERS - 1):
    x, states = tf.keras.layers.RNN(STARCell(RNN_UNITS, t_max=time_dim, dropout=DROPOUT),
                                    return_sequences=True, return_state=True)(x, initial_state=states)

x = tf.keras.layers.RNN(STARCell(RNN_UNITS, t_max=time_dim, dropout=DROPOUT))(x, initial_state=states)

x = tf.keras.layers.Dense(num_classes, activation='linear', bias_initializer='he_uniform')(x)

model = tf.keras.Model(inputs=ip, outputs=x)
model.summary()

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
model.compile(optimizer, loss='mse')

# Train
if not os.path.exists('weights/add/'):
    os.makedirs('weights/add/')

if not os.path.exists('logs/add/'):
    os.makedirs('logs/add/')

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('logs', 'add', timestamp)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('weights/add/model', monitor='val_loss', verbose=1, save_best_only=True),
    tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch='20,50')
]

model.fit(train_dataset, epochs=EPOCHS, verbose=1, callbacks=callbacks,
          validation_data=val_dataset)

# Evaluate model
model = tf.keras.models.load_model('weights/add/model', custom_objects={'STARCell': STARCell})

scores = model.evaluate(test_dataset)

print()
print("Test : ", model.metrics_names, ": ", scores)
