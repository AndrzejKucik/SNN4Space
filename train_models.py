#!/usr/bin/env python3.6

"""Fine tuning of the VGG16 network on the UC Merced or EuroSAT datasets."""

# -- Built-in modules -- #
from argparse import ArgumentParser
import datetime
import os
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_vgg16_model
from dataloaders import load_ucm, load_eurosat
from utils import augment_image, rescale_resize_image, INPUT_FILTER_DICT

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.3.0'
__date__ = '2021-02-05'

# - Argument parser - #
parser = ArgumentParser()
# -- Dataset
parser.add_argument('-ds', '--dataset', type=str, default='ucm',
                    help='Dataset. One of: '
                         + '`eurosat`, `eurosat_prewitt`, `eurosat_prewitt_mask`, `eurosat_sobel`, `eurosat_sobel_mask`'
                         + '`ucm`, `ucm_prewitt`, `ucm_prewitt_mask`, `ucm_sobel`, `ucm_sobel_mask`')
# -- Seed
parser.add_argument('-s', '--seed', type=int, default=5, help='Global random seed.')
# -- Training parameters
parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of training epochs.')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size (per replica).')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate.')
# -- Model parameters
parser.add_argument('-drpt', '--dropout', type=float, default=0, help='Dropout factor. Must be in [0, 1)')
parser.add_argument('-kl2', '--kernel_l2', type=float, default=1e-4,
                    help='Regularization L2 parameter for the convolutional kernels.')
parser.add_argument('-bl1', '--bias_l1', type=float, default=1e-5,
                    help='Regularization L1 parameter for the convolutional biases.')
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
# -- Model parameters
DROPOUT = args['dropout']
KERNEL_L2 = args['kernel_l2']
BIAS_L1 = args['bias_l1']
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
MODEL_FILEPATH = Path('models/vgg16').joinpath(DATASET)
os.makedirs(MODEL_FILEPATH, exist_ok=True)
MODEL_FILEPATH = MODEL_FILEPATH.joinpath('s_{}_e_{}_bs_{}_lr_{}'.format(SEED, EPOCHS, BATCH_SIZE, LR)
                                         + '_drpt_{}_kl2_{}_bl1_{}'.format(DROPOUT, KERNEL_L2, BIAS_L1)
                                         + '_lz_{}_uz_{}'.format(LOWER_ZOOM, UPPER_ZOOM)
                                         + '_mbd_{}_mhd_{}'.format(MAX_BRIGHTNESS_DELTA, MAX_HUE_DELTA)
                                         + '_lc_{}_uc_{}'.format(LOWER_CONTRAST, UPPER_CONTRAST)
                                         + '_ls_{}_us_{}.h5'.format(LOWER_SATURATION, UPPER_SATURATION))


# Preprocessing functions
def rescale_resize(image, label):
    """Rescales and resizes the input images."""

    return rescale_resize_image(image, INPUT_SHAPE[:-1]), label


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
    x_train = x_train.map(rescale_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_val = x_val.map(rescale_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    x_test = x_test.map(rescale_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply random transforms after caching
    x_train = x_train.shuffle(BUFFER_SIZE).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch data
    x_train = x_train.batch(BATCH_SIZE)
    x_val = x_val.batch(BATCH_SIZE)
    x_test = x_test.batch(BATCH_SIZE)

    # Optional gradient-based input (Prewitt and Sobel filters must be applied after batching)
    for key, value in INPUT_FILTER_DICT.items():
        if key in DATASET:
            x_train = x_train.map(value, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            x_val = x_val.map(value, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            x_test = x_test.map(value, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data
    x_train = x_train.prefetch(tf.data.experimental.AUTOTUNE)
    x_val = x_val.prefetch(tf.data.experimental.AUTOTUNE)
    x_test = x_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Compile the model
    with STRATEGY.scope():
        # Create model
        model = create_vgg16_model(input_shape=INPUT_SHAPE,
                                   kernel_l2=KERNEL_L2,
                                   bias_l1=BIAS_L1,
                                   dropout=DROPOUT,
                                   num_classes=NUM_CLASSES)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(LR),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # Show the summary of the model
    model.summary()

    # Callbacks
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 tf.keras.callbacks.ReduceLROnPlateau(verbose=True, patience=50),
                 tf.keras.callbacks.EarlyStopping(patience=100),
                 tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FILEPATH, verbose=True, save_best_only=True)]

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
