#!/usr/bin/env python3.6

"""Estimators of flops, MACs, ACs, and energy consumption of ANN and SNN models during the inference.."""

# -- Built-in modules -- #
from argparse import ArgumentParser

# -- Third-party modules -- #
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from dataloaders import load_eurosat, load_ucm
from utils import rescale_resize_image, INPUT_FILTER_DICT

# -- File info -- #
__author__ = ['Andrzej S. Kucik', 'Gabriele Meoni']
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.2'
__date__ = '2021-02-05'


def number_of_layer_ops(layer, include_bias: bool = False):
    """
    Function calculating the number of operations (addition, multiplication and MAC) within an ANN layer.

    Parameters
    ----------
    layer :
        th.keras.layers.Layer object across which we count the number of operations.
    include_bias : bool
        Whether to account bias addition as an operation (otherwise we assume that the layer initial value was
        initialized with the bias value. It is automatically set to `False` if the layer attribute `use_bias` is
        `False`.

    Returns
    -------
    additions : int
        Number of addition operations performed by the layer.
    multiplications : int
        Number of multiplication operations performed by the layer.
    additions : int
        Number of MAC operations performed by the layer.
    """

    # Input and output tensors
    input_tensor = layer.input
    output_tensor = layer.output

    # Number of the output nodes is the product of the dimensions of the output
    num_output_neurons = np.prod(output_tensor.shape[1:])

    # Config of the layer
    config = layer.get_config()

    # If the network does not use bias, do not take it into account
    try:
        if not config['use_bias']:
            include_bias = False
    except KeyError:
        pass

    if isinstance(layer, tf.keras.layers.Conv2D):
        input_channels = input_tensor.shape[-1] if config['data_format'] == 'channels_last' else input_tensor.shape[1]
        num_input_connections = np.prod(config['kernel_size']) * input_channels
        multiplications = num_output_neurons * num_input_connections
        additions = multiplications + include_bias * num_output_neurons
        macs = multiplications

    elif isinstance(layer, tf.keras.layers.Dense):
        num_input_connections = input_tensor.shape[-1]
        multiplications = num_output_neurons * num_input_connections
        additions = multiplications + include_bias * num_output_neurons
        macs = multiplications

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        num_input_connections = np.prod(config['pool_size'])
        multiplications = num_output_neurons
        additions = num_input_connections * num_output_neurons
        macs = additions

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        if config['data_format'] == 'channels_last':
            num_input_connections = np.prod(input_tensor.shape[1:2])
        else:
            num_input_connections = np.prod(input_tensor.shape[-2:-1])
        multiplications = num_output_neurons
        additions = num_input_connections * num_output_neurons
        macs = additions

    # We assume that all other layer types are just passing the input
    else:
        multiplications, additions, macs = 0, 0, 0

    return additions, multiplications, macs


def number_of_model_ops(model, include_bias: bool = False):
    """
    Function counting the number operations (addition, multiplication and MAC) for an ANN model.

    Parameters
    ----------
    model :
        tf.keras.Model object for which we count the number of operations.
    include_bias : bool
        Whether to account bias addition as an operation (otherwise we assume that the layers initial value were
        initialized with the bias value. It is automatically set to `False` if the layer attribute `use_bias` is
        `False`, for each layer.

    Returns
    -------
    additions : int
        Number of addition operations performed by the model.
    multiplications : int
        Number of multiplication operations performed by the model.
    additions : int
        Number of MAC operations performed by the model.
    """

    # Initialize the operation counter with zeros
    ops = np.zeros((3,), dtype=np.int64)

    # Loop over the model's layers
    for layer in model.layers:
        ops += np.array(number_of_layer_ops(layer, include_bias=include_bias), dtype=np.int64)

    return ops[0], ops[1], ops[2]


def number_of_spikes_model(model, t: int = 1):
    """
    Created a model which counts the number of spikes during the inference.

    Parameters
    ----------
    model :
        tf.keras.Model object of which we count the number of spikes.
    t : int
        Number of timesteps for the spiking model simulation.

    Returns
    -------
    spikes_model :
        tf.keras.Model object counting the number of spikes.
    """

    # Initialize the spikes counter
    sum_of_spikes = None

    # If the first layer in the loaded model is an input layer, we skip it
    n = 0 if isinstance(model.layers[0], tf.keras.layers.Conv2D) else 1

    # Get ne number of spikes from all the other layers
    for layer in model.layers[n:]:
        config = layer.get_config()
        input_spikes = tf.round(t * layer.input)

        if isinstance(layer, tf.keras.layers.Conv2D):
            spikes = tf.keras.layers.Conv2D(trainable=False,
                                            filters=config['filters'],
                                            kernel_size=config['kernel_size'],
                                            strides=config['strides'],
                                            padding=config['padding'],
                                            kernel_initializer='ones',
                                            use_bias=False,
                                            name=config['name'] + '_spikes')(input_spikes)

        elif isinstance(layer, tf.keras.layers.Dense):
            spikes = tf.keras.layers.Dense(trainable=False,
                                           units=config['units'],
                                           kernel_initializer='ones',
                                           use_bias=False,
                                           name=config['name'] + '_spikes')(input_spikes)

        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            spikes = tf.keras.layers.Conv2D(trainable=False,
                                            filters=1,
                                            kernel_size=config['pool_size'],
                                            strides=config['strides'],
                                            padding=config['padding'],
                                            kernel_initializer='ones',
                                            use_bias=False,
                                            name=config['name'] + '_spikes')(input_spikes)

        # - For global average pooling
        else:
            spikes = input_spikes

        # - Sum the number of spikes across the axes which are not the batch normalization axis
        axes_to_reduce = range(1, len(spikes.shape))
        if sum_of_spikes is None:
            sum_of_spikes = tf.reduce_sum(spikes, axis=axes_to_reduce)
        else:
            sum_of_spikes += tf.reduce_sum(spikes, axis=axes_to_reduce)

    # Define the model
    spikes_model = tf.keras.Model(model.input, sum_of_spikes)

    return spikes_model


# noinspection PyUnboundLocalVariable
def main():
    """The main function."""
    # - Argument parser - #
    parser = ArgumentParser()
    parser.add_argument('-md', '--model_path', type=str, default='', required=True, help='Path to the model.')
    parser.add_argument('-ea', '--energy_addition', type=float, default=.1,
                        help='Energy (in pJ) required for performing a single addition operation.')
    parser.add_argument('-if', '--input_filter', type=str, default='', help='Type of the input filter (if any).')
    parser.add_argument('-em', '--energy_multiplication', type=float, default=3.1,
                        help='Energy (in pJ) required for performing a single multiplication operation.')
    parser.add_argument('-t', '--timesteps', type=int, default=1,
                        help='Number of timesteps for the spiking model simulation.')

    args = vars(parser.parse_args())
    path_to_model = args['model_path']
    input_filter = args['input_filter'].lower()
    energy_addition = args['energy_addition']
    energy_multiplication = args['energy_multiplication']
    timesteps = args['timesteps']

    # noinspection PyUnboundLocalVariable
    def energy_consumption(num_ops: int = 0, op_type: str = 'add'):
        """
        Function calculating the energy consumption (in Joules) for a given number of operations.

        Parameters
        ----------
        num_ops : int
            Number of operations.
        op_type : str
            Type of the operation. Must contain `add` for addition, or `mul` for multiplication, or `mac` for MAC.

        Returns
        -------
            energy : float
        """

        if 'add' in op_type.lower():
            p_j = energy_addition
        elif 'mul' in op_type.lower():
            p_j = energy_multiplication
        elif 'mac' in op_type.lower():
            p_j = energy_addition + energy_multiplication
        else:
            exit('Operation type not understood!')

        # Convert to pico Joules to Joules
        energy = 1e-12 * p_j * num_ops

        return energy

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)
    except OSError:
        exit('Invalid model path!')

    # Number of operations
    add, mul, mac = number_of_model_ops(model=model)

    # Spikes model
    # noinspection PyUnboundLocalVariable
    spikes_model = number_of_spikes_model(model=model, t=timesteps)

    # Input and output shapes
    input_shape = model.input.shape[1:]
    num_classes = model.output.shape[-1]

    if input_shape == (64, 64, 3) and num_classes == 10:
        _, _, x_test, labels = load_eurosat()
        num_test = 2700
    elif input_shape == (224, 224, 3) and num_classes == 21:
        _, _, x_test, labels = load_ucm()
        num_test = 210
    else:
        exit("Invalid model!")

    # Preprocessing function
    def rescale_resize(image, label):
        """Rescales and resizes the input images."""

        return rescale_resize_image(image, input_shape[:-1]), label

    # noinspection PyUnboundLocalVariable
    x_test = x_test.map(rescale_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=num_test)
    if input_filter in INPUT_FILTER_DICT.keys():
        x_test = x_test.map(INPUT_FILTER_DICT[input_filter], num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Calculate the number of spikes and average over the batch
    num_spikes = spikes_model.predict(x_test)
    mean_spikes = np.mean(num_spikes)

    # Print results
    # - ANN
    print('ANN:')
    print('Number of additions: {},\t energy: {}mJ'.format(add, 1000 * energy_consumption(add, op_type='add')))
    print('Number of multiplications: {},\t energy: {}mJ'.format(mul, 1000 * energy_consumption(mul, op_type='mul')))
    print('Number of flops: {},\t\t energy: {}mJ'.format(add + mul, 1000 * (energy_consumption(add, op_type='add')
                                                                            + energy_consumption(mul, op_type='mul'))))
    print('Number of MACs: {},\t\t energy: {}mJ\n'.format(mac, 1000 * energy_consumption(mac, op_type='mac')))

    # - SNN
    print('SNN:')
    print('Average number of spikes: {},\t energy: {}mJ'.format(mean_spikes, 1000 * energy_consumption(int(mean_spikes),
                                                                                                       op_type='add')))


if __name__ == '__main__':
    main()
