#!/usr/bin/env python3.6

"""Creates VGG16-based model with (optional) average pooling, and regularizers."""

# -- Third-party modules -- #
import keras_spiking
import numpy as np
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.1'
__date__ = '2021-02-25'


def remove_pooling_kernel(kernel):
    """
    Function transforming the kernel of a 3x3 convolutional layer with strides (1, 1), followed by a 2x2 average
    pooling layer with strides (2, 2) to a kernel of a 4x4 convolutional layer with strides (2, 2), such that
    the resulting output is the same (assuming a piecewise linear activation such as ReLU).

    Parameters
    ----------
    kernel :
        Numpy array with shape (3, 3, input_channels, output_channels).

    Returns
    -------
    kernel :
        Numpy array with shape (4, 4, input_channels, output_channels).
    """

    assert kernel.shape[0:2] == (3, 3)

    new_kernel = np.zeros(shape=(4, 4, kernel.shape[2], kernel.shape[3]))

    new_kernel[:3, :3] += kernel
    new_kernel[:3, 1:] += kernel
    new_kernel[1:, :3] += kernel
    new_kernel[1:, 1:] += kernel

    return new_kernel * .25


def create_vgg16_model(input_shape: tuple = (224, 224, 3),
                       kernel_l2: float = 0.,
                       bias_l1: float = 0.,
                       num_classes: int = 1000,
                       remove_pooling: bool = False,
                       use_dense_bias: bool = False):
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
    num_classes : int
        Number of output classes.
    remove_pooling : bool
        If `True`, then 3x3 stride-1 convolutions with 2x2 pooling layer will be replaced with 4x4 stride-2 convolutions
    use_dense_bias :bool
        If `True`, then bias will be used in the last layer.

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
    for i, layer in enumerate(vgg16.layers[:-1]):
        config = layer.get_config()

        # - Convolutional layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            # -- Add regularizers, if needed
            kernel_regularizer = tf.keras.regularizers.l2(kernel_l2) if kernel_l2 > 0 else None
            bias_regularizer = tf.keras.regularizers.l1(bias_l1) if bias_l1 > 0 else None

            # -- Get weights
            kernel, bias = layer.get_weights()

            # -- Remove pooling, if needed
            if remove_pooling:
                if (isinstance(vgg16.layers[i + 1], tf.keras.layers.AveragePooling2D)
                        or isinstance(vgg16.layers[i + 1], tf.keras.layers.MaxPooling2D)):
                    config['kernel_size'] = (4, 4)
                    config['strides'] = (2, 2)
                    kernel = remove_pooling_kernel(kernel)

            # -- Add the layer to the model
            model.add(tf.keras.layers.Conv2D(filters=config['filters'],
                                             kernel_size=config['kernel_size'],
                                             strides=config['strides'],
                                             padding=config['padding'],
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             activation=tf.nn.relu,
                                             name=config['name'],
                                             input_shape=layer.input.shape[1:]))

            # -- Load weights
            model.layers[-1].set_weights([kernel, bias])

        # - If removing pooling layers is not necessary, then we replace max with average pooling
        if not remove_pooling and isinstance(layer, tf.keras.layers.MaxPooling2D):
            model.add(tf.keras.layers.AveragePooling2D(pool_size=config['pool_size'],
                                                       strides=config['strides'],
                                                       padding=config['padding'],
                                                       name=config['name'],
                                                       input_shape=layer.input_shape))

    # Conclude VGG layers with a global average pooling layer
    model.add(tf.keras.layers.GlobalAveragePooling2D(name='glob_pool'))

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, use_bias=use_dense_bias, name='dense'))

    return model


def create_spiking_vgg16_model(model_path='',
                               input_shape: tuple = (224, 224, 3),
                               dt=0.001,
                               l2: float = 1e-4,
                               lower_hz: float = 10.,
                               upper_hz: float = 20.,
                               tau: float = 0.1,
                               num_classes: int = 1000,
                               spiking_aware_training: bool = True):
    """
    Function returning a spiking version of a VGG16-like model. The input has an additional temporal dimension
    (following the batch size), pooling layers (apart from the final global pooling layer) are removed, and ReLU is
    replaced with spiking activation and low-pass filter.

    Parameters
    ----------
    model_path :
        Path to a pretrained model, if not valid, then VGG weights will be loaded.
    input_shape : tuple
        Shape of the input tensor (with no temporal dimension specified to allow different simulation lengths).
    dt :
        Time resolution of the spiking simulation, either float or tf.Variable, if it is to be decayed.
    l2 : float
        Regularization penalty for the spiking activations.
    lower_hz :float
        Lower spiking frequency desired rate (in Hz) for spiking regularization.
    upper_hz : float
        Lower spiking frequency desired rate (in Hz) for spiking regularization.
    tau :
        Low pass filter parameter tau.
    num_classes : int
        Number of classification classes.
    spiking_aware_training: bool
        If `True`, then the spiking information will be incorporated into the training passes.

    Returns
    -------
    model :
        tf.keras.Model object.
    """

    try:
        # Load a pretrained model from the specified path
        model = tf.keras.models.load_model(model_path)
        dense_load_weights = True
        print('Loaded a pretrained model.')
    except OSError:
        # Load the VGG16 model (this may take a moment for the first time)
        model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        dense_load_weights = False
        print('Loaded the VGG16 model pretrained on ImageNet.')

    # The last layer before the dense classifier and the global pooling layer will not output sequences, so we need to
    # know find its index
    last_layers_idx = len(model.layers) - 1
    if isinstance(model.layers[last_layers_idx], tf.keras.layers.Dense):
        last_layers_idx -= 1
    if isinstance(model.layers[last_layers_idx], tf.keras.layers.GlobalAveragePooling2D):
        last_layers_idx -= 1

    # Define a new model
    new_model = tf.keras.Sequential()

    # Add the temporal dimension
    new_model.add(tf.keras.layers.Reshape((-1,) + input_shape, input_shape=(None,) + input_shape, name='reshape'))

    # Loop over VGG16 layers
    for i, layer in enumerate(model.layers[:last_layers_idx + 1]):
        config = layer.get_config()

        # - Convolutional layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            # -- Get weights
            kernel, bias = layer.get_weights()

            # -- Remove pooling layers, if necessary
            if (isinstance(model.layers[i + 1], tf.keras.layers.AveragePooling2D)
                    or isinstance(model.layers[i + 1], tf.keras.layers.MaxPooling2D)):
                config['kernel_size'] = (4, 4)
                config['strides'] = (2, 2)
                kernel = remove_pooling_kernel(kernel)

            # -- Add the layer to the model
            new_model.add(tf.keras.layers.Conv2D(filters=config['filters'],
                                                 kernel_size=config['kernel_size'],
                                                 strides=config['strides'],
                                                 padding=config['padding'],
                                                 activation=tf.nn.relu,
                                                 name=config['name'],
                                                 input_shape=layer.input.shape[1:]))

            # -- Load weights
            new_model.layers[-1].set_weights([kernel, bias])

            # -- Activation
            activity_regularizer = keras_spiking.regularizers.L2(l2=l2, target=(lower_hz, upper_hz)) if l2 > 0 else None
            # noinspection PyTypeChecker
            new_model.add(keras_spiking.SpikingActivation('relu',
                                                          dt=dt,
                                                          spiking_aware_training=spiking_aware_training,
                                                          activity_regularizer=activity_regularizer,
                                                          name='relu_' + str(i)))

            # -- Low-pass filter. The last filter does not return sequences
            new_model.add(keras_spiking.Lowpass(tau=tau,
                                                dt=dt,
                                                apply_during_training=True,
                                                return_sequences=(i != last_layers_idx - 1),
                                                name='lowpass_' + str(i)))

    #  Conclude VGG layers with a global average pooling layer
    new_model.add(tf.keras.layers.GlobalAveragePooling2D(name='glob_pool'))

    # Output layer
    new_model.add(tf.keras.layers.Dense(num_classes, name='dense'))

    # - Load the final weights, if not using VGG16
    if dense_load_weights:
        weights = model.layers[-1].get_weights()
        # - No bias in the loaded model layer
        if len(weights) == 1:
            kernel = weights[0]
            bias = new_model.layers[-1].get_weights()[-1]
        else:
            kernel, bias = weights

        # - Load the new weights
        new_model.layers[-1].set_weights([kernel, bias])

    return new_model
