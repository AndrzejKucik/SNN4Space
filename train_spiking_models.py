#!/usr/bin/env python3.6

"""Spiking aware fine tuning of the VGG16 network on the UC Merced or EuroSAT datasets."""

# -- Built-in modules -- #
import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_spiking_vgg16_model
from dataloaders import load_ucm, load_eurosat
from utils import augment_image, input_filter_map, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2021-02-19'

# - Argument parser - #
parser = ArgumentParser()
# -- Dataset
parser.add_argument('-ds', '--dataset', type=str, default='ucm',
                    help='Dataset. Either `eurosat` or `ucm`. One can also add `prewitt`, `sobel`, `mask` or `sq`.')
# -- Seed
parser.add_argument('-s', '--seed', type=int, default=5, help='Global random seed.')
# -- Training parameters
parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of training epochs.')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size (per replica).')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate.')
# -- Spike parameters
parser.add_argument('-t', '--timesteps', type=int, default=1, help='Timesteps.')
parser.add_argument('-dt', '--dt', type=float, default=0.001, help='Time delta.')
parser.add_argument('-l2', '--l2', type=float, default=1e-4, help='L2 regularization for the spike frequencies.')
parser.add_argument('-lhz', '--lower_hz', type=float, default=10, help='Lower frequency target for the spikes (Hz).')
parser.add_argument('-uhz', '--upper_hz', type=float, default=20, help='Upper frequency target for the spikes (Hz).')
parser.add_argument('-tau', '--tau', type=float, default=0.1, help='Time delta.')
# -- Augmentation parameters
parser.add_argument('-lz', '--lower_zoom', type=float, default=.95,
                    help='Augmentation parameter. Lower bound for a random zoom factor. Must be positive.')
parser.add_argument('-uz', '--upper_zoom', type=float, default=1.05,
                    help='Augmentation parameter. Upper bound for a random zoom factor. '
                         + 'Must be bigger than lower_zoom.')
parser.add_argument('-mbd', '--max_brightness_delta', type=float, default=.2,
                    help='Augmentation parameter. Maximum brightness delta. Must be a non-negative float.')
parser.add_argument('-mhd', '--max_hue_delta', type=float, default=.1,
                    help='Augmentation parameter. Maximum hue delta. Must be in the interval [0, .5].')
parser.add_argument('-lc', '--lower_contrast', type=float, default=.2,
                    help='Augmentation parameter. Lower bound for a random contrast factor. Must be positive.')
parser.add_argument('-uc', '--upper_contrast', type=float, default=1.8,
                    help='Augmentation parameter. Upper bound for a random contrast factor. '
                         + 'Must be bigger than lower_contrast.')
parser.add_argument('-ls', '--lower_saturation', type=float, default=.9,
                    help='Augmentation parameter. Lower bound for a random saturation factor. Must be positive.')
parser.add_argument('-us', '--upper_saturation', type=float, default=1.1,
                    help='Augmentation parameter. Upper bound for a random saturation factor. '
                         + 'Must be bigger than lower_saturation.')

# - Parse arguments
args = vars(parser.parse_args())
# -- Dataset
DATASET = args['dataset'].lower()
# -- Seed
SEED = args['seed']
# -- Training parameters
EPOCHS = args['epochs']
BATCH_PER_REPLICA = args['batch_size']
LR = args['learning_rate']
# -- Spikes parameters
TIMESTEPS = args['timesteps']
DT = args['dt']
L2 = args['l2']
LOWER_HZ = args['lower_hz']
UPPER_HZ = args['upper_hz']
TAU = args['tau']
# -- Augmentation parameters
LOWER_ZOOM = args['lower_zoom']
UPPER_ZOOM = args['upper_zoom']
MAX_BRIGHTNESS_DELTA = args['max_brightness_delta']
MAX_HUE_DELTA = args['max_hue_delta']
LOWER_CONTRAST = args['lower_contrast']
UPPER_CONTRAST = args['upper_contrast']
LOWER_SATURATION = args['lower_saturation']
UPPER_SATURATION = args['upper_saturation']

# Fix the dataset parameters
if 'eurosat' in DATASET:
    INPUT_SHAPE = (64, 64, 3)
    NUM_CLASSES = 10
    BUFFER_SIZE = 21600
elif 'ucm' in DATASET:
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 21
    BUFFER_SIZE = 1680
else:
    exit('Invalid dataset!')

# Set the seed for reproducibility
tf.random.set_seed(seed=SEED)

# Strategy parameters (for multiple GPU training) #
STRATEGY = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
NUM_DEVICES = STRATEGY.num_replicas_in_sync
print('Number of devices: {}'.format(NUM_DEVICES))

# Global batch size #
BATCH_SIZE = BATCH_PER_REPLICA * NUM_DEVICES

# Model filepath #
MODEL_FILEPATH = Path('models/spiking_vgg16').joinpath(DATASET)
os.makedirs(MODEL_FILEPATH, exist_ok=True)
MODEL_FILEPATH = MODEL_FILEPATH.joinpath('s_{}_e_{}_bs_{}_lr_{}'.format(SEED, EPOCHS, BATCH_SIZE, LR)
                                         + '_dt_{}_l2_{}_lhz_{}_uhz_{}_tau_{}'.format(DT, L2, LOWER_HZ, UPPER_HZ, TAU)
                                         + '_lz_{}_uz_{}'.format(LOWER_ZOOM, UPPER_ZOOM)
                                         + '_mbd_{}_mhd_{}'.format(MAX_BRIGHTNESS_DELTA, MAX_HUE_DELTA)
                                         + '_lc_{}_uc_{}'.format(LOWER_CONTRAST, UPPER_CONTRAST)
                                         + '_ls_{}_us_{}.h5'.format(LOWER_SATURATION, UPPER_SATURATION))


def augment(image, label):
    """Randomly augments the input images."""

    image = augment_image(image=image,
                          image_size=INPUT_SHAPE[:-1],
                          lower_zoom=LOWER_ZOOM,
                          upper_zoom=UPPER_ZOOM,
                          max_brightness_delta=MAX_BRIGHTNESS_DELTA,
                          max_hue_delta=MAX_HUE_DELTA,
                          lower_contrast=LOWER_CONTRAST,
                          upper_contrast=UPPER_CONTRAST,
                          lower_saturation=LOWER_SATURATION,
                          upper_saturation=UPPER_SATURATION)

    return image, label


# Main
def main():
    """The main function."""

    # Load data
    if 'eurosat' in DATASET:
        x_train, x_val, x_test, _ = load_eurosat()
    else:  # 'ucm' in DATASET
        x_train, x_val, x_test, _ = load_ucm()

    # Apply preprocessing functions (no augmentation for the validation and test sets) after caching to avoid caching
    # randomness
    x_train = x_train.map(rescale_resize(image_size=INPUT_SHAPE[:-1]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_val = x_val.map(rescale_resize(image_size=INPUT_SHAPE[:-1]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_test = x_test.map(rescale_resize(image_size=INPUT_SHAPE[:-1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply random transforms after caching
    x_train = x_train.shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch data
    x_train = x_train.batch(BATCH_SIZE)
    x_val = x_val.batch(BATCH_SIZE)
    x_test = x_test.batch(BATCH_SIZE)

    # Optional gradient-based input (Prewitt and Sobel filters must be applied after batching)
    x_train = x_train.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_val = x_val.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data
    x_train = x_train.prefetch(tf.data.experimental.AUTOTUNE)
    x_val = x_val.prefetch(tf.data.experimental.AUTOTUNE)
    x_test = x_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Compile the model
    with STRATEGY.scope():
        dt_var = tf.Variable(1.0)
        # Create model
        model = create_spiking_vgg16_model(input_shape=INPUT_SHAPE,
                                           dt=DT,
                                           l2=L2,
                                           lower_hz=LOWER_HZ,
                                           upper_hz=UPPER_HZ,
                                           tau=TAU,
                                           num_classes=NUM_CLASSES)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(LR),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # Show the summary of the model
    model.summary()

    # Callbacks
    class DTScheduler(tf.keras.callbacks.Callback):
        def __init__(self, dt, scheduler):
            super(DTScheduler, self).__init__()
            self.dt = dt
            self.scheduler = scheduler
            self.step = 0

        def on_train_batch_begin(self, batch, logs=None):
            self.dt.assign(self.scheduler.__call__(step=self.step))
            self.step += 1

    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 DTScheduler(dt=dt_var,
                             scheduler=tf.keras.optimizers.schedules.ExponentialDecay(1.0,
                                                                                      decay_steps=EPOCHS,
                                                                                      decay_rate=DT ** (1 / EPOCHS),
                                                                                      staircase=False)),
                 tf.keras.callbacks.ReduceLROnPlateau(patience=50, verbose=True),
                 tf.keras.callbacks.EarlyStopping(patience=100, verbose=True),
                 tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FILEPATH, save_best_only=True, verbose=True, )]

    # Train the model
    model.fit(x=x_train, epochs=EPOCHS, validation_data=x_val, callbacks=callbacks)

    # Load the best weights
    model.load_weights(MODEL_FILEPATH)

    # Evaluate the model
    loss, acc = model.evaluate(x=x_test)
    print('Best model\'s accuracy: {:.2f}%.'.format(acc * 100))

    # Rename the best model file to include the accuracy in its name
    os.rename(MODEL_FILEPATH, MODEL_FILEPATH.with_name(MODEL_FILEPATH.stem + '_acc_{:.2f}.h5'.format(acc * 100)))


if __name__ == '__main__':
    main()
