#!/usr/bin/env python3.6

"""Creates VGG16-based model with average pooling, dropout, and regularizers."""

# -- Third-party modules -- #
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2021-02-01'


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
    model
        tf.keras.Model object.
    """

    # Load the VGG16 model (this may take a moment for the first time)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape,
                                        pooling='avg')

    # Create new model
    model = tf.keras.Sequential()

    # Add layers
    for layer in vgg16.layers:
        # - Add regularization to the convolutional layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            if kernel_l2 > 0:
                setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(kernel_l2))
            if bias_l1 > 0:
                setattr(layer, 'bias_regularizer', tf.keras.regularizers.l1(bias_l1))
            model.add(layer)

        # - Swap max for average pooling
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            model.add(tf.keras.layers.AveragePooling2D((2, 2)))
            # - Add dropout, if needed
            if dropout > 0.:
                model.add(tf.keras.layers.Dropout(dropout))
        else:
            model.add(layer)

    # Final dense layer
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))

    return model


def create_resnet50_model():
    # TO DO

    pass
