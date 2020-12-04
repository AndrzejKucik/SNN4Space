#!/usr/bin/env python3.6

"""Fine tuning of the VGG16 network on the UC Merced dataset."""

# -- Built-in modules -- #
import datetime

# -- Third-party modules -- #
import tensorflow as tf
import tensorflow_datasets as tfds

# -- Proprietary modules -- #
from utils import augment, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2020-12-04'

# - Parameters - #
SEED = 5
tf.random.set_seed(seed=SEED)  # Set the seed for reproducibility

# -- Strategy parameters (for multiple GPU training) -- #
STRATEGY = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
NUM_DEVICES = STRATEGY.num_replicas_in_sync
print('Number of devices: {}'.format(NUM_DEVICES))

# -- Shape parameters -- #
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 21

# -- Augmentation parameters -- #
MAX_BRIGHTNESS_DELTA = .2
MAX_HUE_DELTA = .1
LOWER_CONTRAST = .2
UPPER_CONTRAST = 1.8
LOWER_SATURATION = .9
UPPER_SATURATION = 1.1

# -- Straining parameters -- #
EPOCHS = 1000
BATCH_PER_REPLICA = 105
BATCH_SIZE = BATCH_PER_REPLICA * NUM_DEVICES
LR = .001

# -- Model parameters -- #
DROPOUT = 0.25
KERNEL_L2 = 1e-4
BIAS_L1 = 1e-5
MODEL_FILEPATH = './models/seed_{}_epochs_{}_lr_{}_dropout_{}_kl2_{}_bl1_{}.h5'.format(SEED, EPOCHS, LR, DROPOUT,
                                                                                       KERNEL_L2, BIAS_L1)


# Preprocessing functions
def preprocess(image, label):
    """Rescales and resizes the input images."""

    return rescale_resize(image, INPUT_SHAPE[:-1]), label


def preprocess_with_aug(image, label):
    """Rescales, resizes and augments the input images."""

    image = augment(image=image,
                    image_size=INPUT_SHAPE[:-1],
                    max_brightness_delta=MAX_BRIGHTNESS_DELTA,
                    max_hue_delta=MAX_HUE_DELTA,
                    lower_contrast=LOWER_CONTRAST,
                    upper_contrast=UPPER_CONTRAST,
                    lower_saturation=LOWER_SATURATION,
                    upper_saturation=UPPER_SATURATION)

    return image, label


# For visualization
def postprocess(image, label):
    """Rescales and resizes the input images (for visualization)."""

    return tf.cast(image * 255., tf.uint8), label


def load_data(show_examples=False):
    """Loads and preprocesses the UC Merced dataset as Tensorflow datasets."""

    # The UC Merced dataset (this takes a while for the first time)
    (ucm_train, ucm_val, ucm_test), info = tfds.load('uc_merced',
                                                     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                     shuffle_files=True,
                                                     with_info=True,
                                                     as_supervised=True)
    num_train = int(info.splits['train'].num_examples * .8)
    num_val = int(info.splits['train'].num_examples * .1)
    num_test = int(info.splits['train'].num_examples * .1)
    print('Number of training examples: {}, validation examples: {}, and test examples: {}'.format(num_train,
                                                                                                   num_val,
                                                                                                   num_test))
    if show_examples:
        # Visualize the training data before augmentation
        tfds.visualization.show_examples(ds=ucm_train, ds_info=info)

    # Apply preprocessing functions (no augmentation for the validation and test sets)
    ucm_train = ucm_train.map(preprocess_with_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(num_train)
    ucm_val = ucm_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ucm_test = ucm_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if show_examples:
        # Visualize the training data after augmentation
        tfds.visualization.show_examples(
            ds=ucm_train.map(postprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE), ds_info=info)

    return (ucm_train, ucm_test, ucm_val), info


def create_model(input_shape: tuple = INPUT_SHAPE,
                 num_classes: int = NUM_CLASSES,
                 kernel_l2: float = KERNEL_L2,
                 bias_l1: float = BIAS_L1,
                 dropout: float = DROPOUT):
    """Creates a Keras model which is a modified version of the VGG16 network."""

    # The VGG model (this can take a moment for the first time)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    with STRATEGY.scope():
        # Create new model
        # - Input layer
        input_layer = tf.keras.Input(shape=input_shape)

        # - First convolutional layer
        config = vgg16.layers[1].get_config()
        filters, kernel_size, name = config['filters'], config['kernel_size'], config['name']
        # -- We keep the same number of filters and the kernel size but change the activation format and add
        # -- kernel and bias regularizers
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
                                   bias_regularizer=tf.keras.regularizers.l1(bias_l1),
                                   name=name)(input_layer)

        # - The remaining layers
        for layer in vgg16.layers[2:-1]:
            try:
                config = layer.get_config()
                filters, kernel_size, name = config['filters'], config['kernel_size'], config['name']
                x = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           padding='same',
                                           activation=tf.nn.relu,
                                           kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
                                           bias_regularizer=tf.keras.regularizers.l1(bias_l1),
                                           name=name)(x)
            except KeyError:
                # -- Change the max pooling to average pooling
                x = tf.keras.layers.AveragePooling2D((2, 2))(x)
                # -- Add dropout if necessary
                if DROPOUT > 0.:
                    x = tf.keras.layers.Dropout(dropout)(x)

        # - Conclude VGG layers with a global average pooling layer
        global_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

        # - Output layer
        output_layer = tf.keras.layers.Dense(num_classes, use_bias=False)(global_pool)

        # - Define the model
        model = tf.keras.Model(input_layer, output_layer)

        # - After the model is defined, we can load the weights
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = vgg16.get_layer(name=layer.name).get_weights()
                layer.set_weights(weights)

    return model


def train_model(model, train_data, val_data, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, lr: float = LR):
    """Trains the model."""

    # Prepare the data
    train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Callbacks
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 tf.keras.callbacks.ReduceLROnPlateau(verbose=True, patience=50),
                 tf.keras.callbacks.EarlyStopping(patience=100),
                 tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FILEPATH, verbose=True, save_best_only=True)]

    with STRATEGY.scope():
        # - Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # Train the model
    model.fit(x=train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    # Load the best weights
    model.load_weights(MODEL_FILEPATH)


# Main
def main():
    """The main function."""
    # Load data
    (ucm_train, ucm_val, ucm_test), info = load_data(show_examples=True)

    # Create model
    model = create_model()

    # Show the summary of the model
    model.summary()

    # Train model
    train_model(model=model, train_data=ucm_train, val_data=ucm_test)

    # Evaluate the model
    loss, acc = model.evaluate(x=ucm_test.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))
    print('Best model\'s accuracy: {:2f}%.'.format(acc * 100))


if __name__ == '__main__':
    main()
