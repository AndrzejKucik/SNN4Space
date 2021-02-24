#!/usr/bin/env python3.6

"""Fine tuning of the VGG16 network on the UC Merced or EuroSAT datasets."""

# -- Built-in modules -- #
import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_vgg16_model
from dataloaders import load_ucm, load_eurosat
from utils import augment, input_filter_map, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.3.2'
__date__ = '2021-02-24'

# - Argument parser - #
parser = ArgumentParser()
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
# -- Training parameters
parser.add_argument('-e',
                    '--epochs',
                    type=int,
                    default=1000,
                    help='Number of training epochs.')
parser.add_argument('-bs',
                    '--batch_size',
                    type=int,
                    default=32,
                    help='Batch size (per replica).')
parser.add_argument('-lr',
                    '--learning_rate',
                    type=float,
                    default=0.001,
                    help='Learning rate.')
# -- Model parameters
parser.add_argument('-kl2',
                    '--kernel_l2',
                    type=float,
                    default=1e-4,
                    help='Regularization L2 parameter for the convolutional kernels.')
parser.add_argument('-bl1',
                    '--bias_l1',
                    type=float,
                    default=1e-5,
                    help='Regularization L1 parameter for the convolutional biases.')
# -- Augmentation parameters
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
                    type=float, default=.2,
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
                    type=float,
                    default=1.1,
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

# Model filepath #
MODEL_FILEPATH = Path('models/vgg16').joinpath(DATASET)
os.makedirs(MODEL_FILEPATH, exist_ok=True)
MODEL_FILEPATH = MODEL_FILEPATH.joinpath('s_{}'.format(SEED)
                                         + '_e_{}'.format(EPOCHS)
                                         + '_bs_{}'.format(BATCH_SIZE)
                                         + '_lr_{}'.format(LR)
                                         + '_kl2_{}'.format(KERNEL_L2)
                                         + '_bl1_{}'.format(BIAS_L1)
                                         + '_lz_{}'.format(args['lower_zoom'])
                                         + '_uz_{}'.format(args['upper_zoom'])
                                         + '_mbd_{}'.format(args['max_brightness_delta'])
                                         + '_mhd_{}'.format(args['max_hue_delta'])
                                         + '_lc_{}'.format(args['lower_contrast'])
                                         + '_uc_{}'.format(args['upper_contrast'])
                                         + '_ls_{}'.format(args['lower_saturation'])
                                         + '_us_{}.h5'.format(args['upper_saturation']))


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

    # - Batch data
    x_train = x_train.batch(BATCH_SIZE)
    x_val = x_val.batch(BATCH_SIZE)
    x_test = x_test.batch(BATCH_SIZE)

    # - Optional gradient-based input (Prewitt and Sobel filters must be applied after batching)
    x_train = x_train.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_val = x_val.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Prefetch data
    x_train = x_train.prefetch(tf.data.experimental.AUTOTUNE)
    x_val = x_val.prefetch(tf.data.experimental.AUTOTUNE)
    x_test = x_test.prefetch(tf.data.experimental.AUTOTUNE)

    # The model
    with STRATEGY.scope():
        # - Create a model
        model = create_vgg16_model(input_shape=INPUT_SHAPE,
                                   kernel_l2=KERNEL_L2,
                                   bias_l1=BIAS_L1,
                                   num_classes=NUM_CLASSES,
                                   remove_pooling=False,
                                   use_dense_bias=False)

        # - Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(LR),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # - Show the summary of the model
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
    print('\nBest model\'s accuracy: {:.2f}%.\n'.format(acc * 100))

    # Rename the best model file to include the accuracy in its name
    save_path = MODEL_FILEPATH.with_name(MODEL_FILEPATH.stem + '_acc_{:.2f}.h5'.format(acc * 100))
    print('\nSaving the model to:' + str(save_path))
    os.rename(MODEL_FILEPATH, save_path)


if __name__ == '__main__':
    main()
