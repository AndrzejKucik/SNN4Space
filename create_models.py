#!/usr/bin/env python3.6

"""Creates VGG16-based model with average pooling, dropout, and regularizers."""

# -- Third-party modules -- #
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.1'
__date__ = '2021-02-03'


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
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')

    # Define a new model
    model = tf.keras.Sequential()

    # Loop over VGG16 layers
    for layer in vgg16.layers:
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


def create_resnet50_model():
    # TO DO

    pass
