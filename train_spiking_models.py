#!/usr/bin/env python3.6

"""Spiking aware fine tuning of the VGG16 network trained on the EuroSAT RGB or UC Merced datasets."""

# -- Built-in modules -- #
import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

# -- Third-party modules -- #
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_spiking_vgg16_model
from dataloaders import load_ucm, load_eurosat
from utils import add_temporal_dim, augment, DTStop, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.2'
__date__ = '2021-02-25'

# - Argument parser - #
parser = ArgumentParser()
# -- Path to a pretrained model
parser.add_argument('-md',
                    '--model_path',
                    required=True,
                    type=str,
                    help='Path to a pretrained model. Make sure that it is consistent with the dataset.')
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
# -- Since the training is done iteratively, they will either increase or decrease by a factor of 2.
parser.add_argument('-e',
                    '--epochs',
                    type=int,
                    default=16,
                    help='Target number of epochs.')
parser.add_argument('-bs',
                    '--batch_size',
                    type=int,
                    default=16,
                    help='Target batch size (per replica).')
parser.add_argument('-lr',
                    '--learning_rate',
                    type=float,
                    default=3e-5,
                    help='Target learning rate.')
# -- Spiking parameters
parser.add_argument('-t',
                    '--timesteps',
                    type=int,
                    default=2,
                    help='Target number of simulation timesteps (the training starts with 1).')
parser.add_argument('-dt',
                    '--dt',
                    type=float,
                    default=1.,
                    help='Starting time resolution for the simulation, this will be decayed during the training.')
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
parser.add_argument('-tau',
                    '--tau',
                    type=float,
                    default=0.1,
                    help='Tau parameter for the low-pass filter.')
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
MODEL_PATH = args['model_path']
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
TAU = args['tau']

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

# Get the exponents for scaling by a factor of 2.
EXPONENT = int(np.log2(TIMESTEPS))


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

    # The model
    with STRATEGY.scope():
        # - A variable to be passed to the model as the simulation time resolution
        dt_var = tf.Variable(DT, aggregation=tf.VariableAggregation.MEAN)

        # - Functions to monitor the variables
        # noinspection PyUnusedLocal
        def dt_monitor(y_true, y_pred):
            return dt_var.read_value()

        # - Create a model
        model = create_spiking_vgg16_model(model_path=MODEL_PATH,
                                           input_shape=INPUT_SHAPE,
                                           dt=dt_var,
                                           l2=L2,
                                           lower_hz=LOWER_HZ,
                                           upper_hz=UPPER_HZ,
                                           tau=TAU,
                                           num_classes=NUM_CLASSES,
                                           spiking_aware_training=True)

        # - Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(LR),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy(), dt_monitor])

    # Show model's summary
    model.summary()

    # Iterate the training, by decreasing the batch size and the learning rate by a power of 2, and increasing the
    # number of timesteps (also by a power of 2), until they reach the target size
    for n in range(EXPONENT + 1):
        # The data
        # - Add the temporal dimension
        timesteps = 2 ** n
        x_train_t = x_train.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x_val_t = x_val.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x_test_t = x_test.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # - Batch the data
        batch_size = 2 ** (EXPONENT - n) * BATCH_SIZE
        x_train_t = x_train_t.batch(batch_size=batch_size)
        x_val_t = x_val_t.batch(batch_size=batch_size)
        x_test_t = x_test_t.batch(batch_size=batch_size)

        # - Prefetch data
        x_train_t = x_train_t.prefetch(tf.data.experimental.AUTOTUNE)
        x_val_t = x_val_t.prefetch(tf.data.experimental.AUTOTUNE)
        x_test_t = x_test_t.prefetch(tf.data.experimental.AUTOTUNE)

        # Learning rate
        lr = LR * 2 ** (EXPONENT - n)
        tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

        # Epochs
        epochs = EPOCHS * 2 ** (EXPONENT - n)

        # Callbacks
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=epochs // 4, verbose=True),
                     DTStop(dt=dt_var, dt_min=.001),
                     tf.keras.callbacks.EarlyStopping(monitor='dt_monitor',
                                                      min_delta=0.001,
                                                      patience=epochs // 4,
                                                      mode='min',
                                                      verbose=True)]

        # Print the training iteration parameters
        print('\nStarting the training for {} epoch(s),'.format(epochs),
              'with {} timestep(s)'.format(timesteps),
              'on batches of {} example(s),'.format(batch_size),
              'and the learning rate {}.'.format(LR * 2 ** (EXPONENT - n)),
              '\nLearning rate reduced after {} epoch(s) of no improvement in the validation loss,'.format(epochs // 4),
              'early stopping after {} epoch(s) of no decay of the dt value.\n'.format(epochs // 4))

        # Train the model
        print('Commencing the training on iteration {}/{}.'.format(n + 1, EXPONENT))
        model.fit(x=x_train_t, epochs=epochs, validation_data=x_val_t, callbacks=callbacks)

        # Evaluate the model
        loss, acc, dt_stop = model.evaluate(x=x_test_t, batch_size=batch_size, verbose=True)
        print('\nModel\'s accuracy: {:.2f}%.\n'.format(acc * 100))

        # New model to avoid serialization issued
        with STRATEGY.scope():
            new_model = create_spiking_vgg16_model(model_path=MODEL_PATH,
                                                   input_shape=INPUT_SHAPE,
                                                   dt=dt_stop,
                                                   l2=L2,
                                                   lower_hz=LOWER_HZ,
                                                   upper_hz=UPPER_HZ,
                                                   tau=TAU,
                                                   num_classes=NUM_CLASSES,
                                                   spiking_aware_training=True)

            # - Load weights
            new_model.set_weights(model.get_weights())

            # - Compile the model
            new_model.compile(optimizer=tf.keras.optimizers.RMSprop(LR),
                              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=[tf.metrics.SparseCategoricalAccuracy()])

        # Save model filepath
        model_filepath = Path('models/spiking_vgg16').joinpath(DATASET)
        os.makedirs(model_filepath, exist_ok=True)
        model_filepath = model_filepath.joinpath('s_{}'.format(SEED)
                                                 + '_e_{}'.format(epochs)
                                                 + '_bs_{}'.format(batch_size)
                                                 + '_lr_{}'.format(lr)
                                                 + '_t_{}'.format(timesteps)
                                                 + '_dt_{}'.format(dt_stop)
                                                 + '_l2_{}'.format(timesteps, L2)
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
                                                 + '_acc_{:.2f}'.format(acc))

        # Save model
        print('\nSaving the model to:' + str(model_filepath))
        new_model.save(model_filepath)

        # We stop optimising dt here
        if dt_stop <= 0.001:
            model = new_model

        del new_model


if __name__ == '__main__':
    main()
