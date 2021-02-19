#!/usr/bin/env python3.6

"""Creates VGG16-based model with average pooling, dropout, and regularizers."""

# -- Third-party modules -- #
import keras_spiking
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.2'
__date__ = '2021-02-19'


def create_vgg16_model(input_shape: tuple = (224, 224, 3),
                       kernel_l2: float = 0.,
                       bias_l1: float = 0.,
                       dropout: float = 0.,
                       num_classes: int = 1000):
    """
    Creates a Keras model which is a modified version of the VGG16 network.

    Parameters
    ----------
    input_shape : tuple
        Tuple of integers (height, width, channels) specifying the shape of the input images.
    kernel_l2 : float
        Weight penalty for convolutional kernels' L2 regularizer. Only applied if positive.
    bias_l1 : float
        weight penalty for bias L1 regularizer. Only applied if positive.
    dropout : float
        Dropout factor. Only applied if positive.
    num_classes : int
        Number of output classes

    Returns
    -------
    model :
        tf.keras.Model object.
    """

    # Load the VGG16 model (this may take a moment for the first time)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # Define a new model
    model = tf.keras.Sequential()

    # Loop over VGG16 layers
    for layer in vgg16.layers[:-1]:
        config = layer.get_config()

        if isinstance(layer, tf.keras.layers.Conv2D):
            # - Add regularizers if needed
            kernel_regularizer = tf.keras.regularizers.l2(kernel_l2) if kernel_l2 > 0 else None
            bias_regularizer = tf.keras.regularizers.l1(bias_l1) if bias_l1 > 0 else None

            model.add(tf.keras.layers.Conv2D(filters=config['filters'],
                                             kernel_size=config['kernel_size'],
                                             strides=config['strides'],
                                             padding=config['padding'],
                                             activation=tf.nn.relu,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             name=config['name'],
                                             input_shape=layer.input.shape[1:]))
            model.layers[-1].set_weights(layer.get_weights())

        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            # - Change the max pooling to average pooling
            model.add(tf.keras.layers.AveragePooling2D(pool_size=config['pool_size'],
                                                       strides=config['strides'],
                                                       padding=config['padding']))
            # -- Add dropout if necessary
            if dropout > 0.:
                model.add(tf.keras.layers.Dropout(dropout))

    # - Conclude VGG layers with a global average pooling layer
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # - Output layer
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))

    return model


def create_spiking_vgg16_model(input_shape: tuple = (224, 224, 3),
                               dt=0.001,
                               l2: float = 1e-4,
                               lower_hz: float = 10.,
                               upper_hz: float = 10.,
                               tau: float = 0.1,
                               num_classes: int = 1000):
    # Load the VGG16 model (this may take a moment for the first time)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # Define a new model
    model = tf.keras.Sequential()

    # Add the temporal dimension
    model.add(tf.keras.layers.Reshape((-1,) + input_shape, input_shape=(None,) + input_shape))

    # Loop over VGG16 layers
    for i, layer in enumerate(vgg16.layers[:-1]):
        config = layer.get_config()

        if isinstance(layer, tf.keras.layers.Conv2D):
            model.add(tf.keras.layers.Conv2D(filters=config['filters'],
                                             kernel_size=config['kernel_size'],
                                             strides=config['strides'],
                                             padding=config['padding'],
                                             name=config['name']))
            model.layers[-1].set_weights(layer.get_weights())

            # noinspection PyTypeChecker
            model.add(keras_spiking.SpikingActivation('relu',
                                                      dt=dt,
                                                      spiking_aware_training=True,
                                                      activity_regularizer=keras_spiking.regularizers.L2(
                                                          l2=l2, target=(lower_hz, upper_hz))))
            # - Last filter does not return sequences
            model.add(keras_spiking.Lowpass(tau=tau, dt=dt, return_sequences=(i != 17)))

        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            # - Change the strides of the last convolution to 2, and divide bias by 4
            model.layers[-3].__setattr__('strides', 2)
            kernel, bias = model.layers[-3].get_weights()
            model.layers[-3].set_weights([kernel, bias / 4])

    # - Conclude VGG layers with a global average pooling layer
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    # - Output layer
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))

    return model


def create_resnet50_model():
    # TO DO

    pass
