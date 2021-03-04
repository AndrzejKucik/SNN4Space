#!/usr/bin/env python3.6

"""Fine-tunes a spiking neural network on fixed temporal resolution."""

# -- Built-in modules -- #
import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_spiking_vgg16_model
from dataloaders import load_eurosat, load_ucm
from utils import add_temporal_dim, augment, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2021-03-04'

# - Argument parser - #
parser = ArgumentParser()
# -- Path to a pretrained model
parser.add_argument('-wp',
                    '--weights_path',
                    required=True,
                    type=str,
                    help='Path to weights of a pretrained SNN.')
# -- Dataset
parser.add_argument('-ds',
                    '--dataset',
                    required=True,
                    type=str,
                    default='eurosat',
                    help='Dataset. Either `eurosat` or `ucm`. One can also add `prewitt`, `sobel`, `mask` or `sq`.')
# -- Seed
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=5,
                    help='Global random seed.')
# -- Training parameters.
parser.add_argument('-e',
                    '--epochs',
                    type=int,
                    default=16,
                    help='Number of training epochs.')
parser.add_argument('-bs',
                    '--batch_size',
                    type=int,
                    default=32,
                    help='Batch size (per replica).')
parser.add_argument('-lr',
                    '--learning_rate',
                    type=float,
                    default=1e-4,
                    help='Learning rate.')
# -- Spiking parameters
parser.add_argument('-t',
                    '--timesteps',
                    type=int,
                    default=50,
                    help='Simulation timesteps.')
parser.add_argument('-dt',
                    '--dt',
                    type=float,
                    default=.001,
                    help='Time resolution for the simulation.')
parser.add_argument('-l2',
                    '--l2',
                    type=float,
                    default=1e-9,
                    help='L2 regularization for the spike frequencies.')
parser.add_argument('-lhz',
                    '--lower_hz',
                    type=float,
                    default=10.,
                    help='Lower frequency target for the spikes (Hz).')
parser.add_argument('-uhz',
                    '--upper_hz',
                    type=float,
                    default=20.,
                    help='Upper frequency target for the spikes (Hz).')
# -- Augmentation parameters (the same as for ANN training).
parser.add_argument('-lz',
                    '--lower_zoom',
                    type=float,
                    default=.95,
                    help='Augmentation parameter. Lower bound for a random zoom factor. Must be positive.')
parser.add_argument('-uz',
                    '--upper_zoom',
                    type=float,
                    default=1.05,
                    help='Augmentation parameter. Upper bound for a random zoom factor. '
                         + 'Must be bigger than lower_zoom.')
parser.add_argument('-mbd',
                    '--max_brightness_delta',
                    type=float,
                    default=.2,
                    help='Augmentation parameter. Maximum brightness delta. Must be a non-negative float.')
parser.add_argument('-mhd',
                    '--max_hue_delta',
                    type=float,
                    default=.1,
                    help='Augmentation parameter. Maximum hue delta. Must be in the interval [0, .5].')
parser.add_argument('-lc',
                    '--lower_contrast',
                    type=float,
                    default=.2,
                    help='Augmentation parameter. Lower bound for a random contrast factor. Must be positive.')
parser.add_argument('-uc',
                    '--upper_contrast',
                    type=float,
                    default=1.8,
                    help='Augmentation parameter. Upper bound for a random contrast factor. '
                         + 'Must be bigger than lower_contrast.')
parser.add_argument('-ls',
                    '--lower_saturation',
                    type=float,
                    default=.9,
                    help='Augmentation parameter. Lower bound for a random saturation factor. Must be positive.')
parser.add_argument('-us',
                    '--upper_saturation',
                    type=float, default=1.1,
                    help='Augmentation parameter. Upper bound for a random saturation factor. '
                         + 'Must be bigger than lower_saturation.')

# -- Parse arguments
args = vars(parser.parse_args())
# --- Path to model
WEIGHTS_PATH = args['weights_path']
# --- Dataset
DATASET = args['dataset'].lower()
# --- Seed
SEED = args['seed']
# --- Training parameters
EPOCHS = args['epochs']
BATCH_PER_REPLICA = args['batch_size']
LR = args['learning_rate']
# --- Spikes parameters
TIMESTEPS = args['timesteps']
DT = args['dt']
L2 = args['l2']
LOWER_HZ = args['lower_hz']
UPPER_HZ = args['upper_hz']

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
print('\nNumber of devices: {}\n'.format(NUM_DEVICES))

# Global batch size #
BATCH_SIZE = BATCH_PER_REPLICA * NUM_DEVICES


# Main
def main():
    """The main function."""

    # Prepare the data for training
    # - Load data
    if 'eurosat' in DATASET:
        x_train, x_val, x_test, _ = load_eurosat()
    else:  # 'ucm' in DATASET
        x_train, x_val, x_test, _ = load_ucm()

    # - Resize and rescale the images, and cache the training and the validation sets for faster training
    x_train = x_train.map(rescale_resize(image_size=INPUT_SHAPE[:-1]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_val = x_val.map(rescale_resize(image_size=INPUT_SHAPE[:-1]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_test = x_test.map(rescale_resize(image_size=INPUT_SHAPE[:-1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Shuffle the training set and apply the augmentation (after caching to avoid caching randomness)
    x_train = x_train.shuffle(BUFFER_SIZE).map(augment(image_size=INPUT_SHAPE[:-1], augmentation_parameters=args),
                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Add the temporal dimension
    x_train = x_train.map(add_temporal_dim(timesteps=TIMESTEPS), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_val = x_val.map(add_temporal_dim(timesteps=TIMESTEPS), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(add_temporal_dim(timesteps=TIMESTEPS), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Batch the data
    x_train = x_train.batch(batch_size=BATCH_SIZE)
    x_val = x_val.batch(batch_size=BATCH_SIZE)
    x_test = x_test.batch(batch_size=BATCH_SIZE)

    # - Prefetch data
    x_train = x_train.prefetch(tf.data.experimental.AUTOTUNE)
    x_val = x_val.prefetch(tf.data.experimental.AUTOTUNE)
    x_test = x_test.prefetch(tf.data.experimental.AUTOTUNE)

    # The model
    with STRATEGY.scope():
        # - Create the model
        model = create_spiking_vgg16_model(input_shape=INPUT_SHAPE,
                                           dt=DT,
                                           l2=L2,
                                           lower_hz=LOWER_HZ,
                                           upper_hz=UPPER_HZ,
                                           num_classes=NUM_CLASSES)

        # - Load the weights
        model.load_weights(filepath=WEIGHTS_PATH)

        # - Compile the model
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(LR,
                                                                       decay_steps=BUFFER_SIZE // BATCH_SIZE,
                                                                       decay_rate=0.9,
                                                                       staircase=True)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # - Model summary
    model.summary()

    # Callbacks
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    # Train the model
    model.fit(x=x_train, epochs=EPOCHS, validation_data=x_val, callbacks=callbacks)

    # Evaluate the model
    loss, acc, = model.evaluate(x=x_test, batch_size=BATCH_SIZE, verbose=True)
    print('\nModel\'s accuracy: {:.2f}%.\n'.format(acc * 100))

    # Save model filepath
    model_filepath = Path('models/spiking_vgg16/fine_tuned').joinpath(DATASET)
    os.makedirs(model_filepath, exist_ok=True)
    model_filepath = model_filepath.joinpath('s_{}'.format(SEED)
                                             + '_e_{}'.format(EPOCHS)
                                             + '_bs_{}'.format(BATCH_SIZE)
                                             + '_lr_{}'.format(LR)
                                             + '_t_{}'.format(TIMESTEPS)
                                             + '_dt_{:.4f}'.format(DT)
                                             + '_l2_{}'.format(L2)
                                             + '_lhz_{}'.format(LOWER_HZ)
                                             + '_uhz_{}'.format(UPPER_HZ)
                                             + '_lz_{}'.format(args['lower_zoom'])
                                             + '_uz_{}'.format(args['upper_zoom'])
                                             + '_mbd_{}'.format(args['max_brightness_delta'])
                                             + '_mhd_{}'.format(args['max_hue_delta'])
                                             + '_lc_{}'.format(args['lower_contrast'])
                                             + '_uc_{}'.format(args['upper_contrast'])
                                             + '_ls_{}'.format(args['lower_saturation'])
                                             + '_us_{}'.format(args['upper_saturation'])
                                             + '_acc_{:.2f}.h5'.format(acc))

    # Save model
    print('\nSaving model weights to: ' + str(model_filepath))
    model.save_weights(model_filepath)


if __name__ == '__main__':
    main()
