#!/usr/bin/env python3.6

"""
Converts a Tensorflow model trained on the EuroSAT or UC Merced dataset into a Nengo spiking neural network and
evaluates it.
"""

# -- Built-in modules -- #
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
from time import time

# -- Third-party modules -- #
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from dataloaders import load_eurosat, load_ucm
from utils import input_filter_map, plot_spikes, rescale_resize

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__contributor__ = 'Gabriele Meoni'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.6'
__date__ = '2021-02-15'

# - Assertions to ensure modules compatibility - #
assert nengo.__version__ == '3.1.0', 'Nengo version is {}, and it should be 3.1.0 instead.'.format(nengo.__version__)
assert nengo_dl.__version__ == '3.4.0', 'NengoDL version is {}, and it should be 3.4.0'.format(nengo_dl.__version__)

color_dictionary = {'red': '\033[0;31m',
                    'black': '\033[0m',
                    'green': '\033[0;32m',
                    'orange': '\033[0;33m',
                    'purple': '\033[0;35m',
                    'blue': '\033[0;34m',
                    'cyan': '\033[0;36m'}

# - Parameters - #
N_NEURONS = 1000
N_EXAMPLES = 10

# - Argument parser - #
parser = ArgumentParser()
parser.add_argument('-md', '--model_path', type=str, required=True, help='Path to the model.')
parser.add_argument('-if', '--input_filter', type=str, default='', help='Type of the input filter (if any).')
parser.add_argument('-sc', '--firing_rate_scale', type=float, default=1, help='scale factor for the firing rate.')
parser.add_argument('-syn', '--synapse', type=float, default=None, help='Value of the synapse.')
parser.add_argument('-t', '--timesteps', type=int, default=1, help='Simulation timesteps.')

args = vars(parser.parse_args())
path_to_model = Path(args['model_path'])
input_filter = args['input_filter'].lower()
scale = args['firing_rate_scale']
synapse = args['synapse']
timesteps = args['timesteps']


# noinspection PyUnboundLocalVariable
def main():
    """The main function."""

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)
    except OSError:
        exit('Invalid model path!')

    # Paramters
    # noinspection PyUnboundLocalVariable
    input_shape = model.input.shape[1:]
    num_classes = model.output.shape[-1]

    # Different dataset parameters
    if input_shape == (64, 64, 3) and num_classes == 10:
        dataset = 'eurosat'
        print('Using', color_dictionary['red'], 'EuroSAT', color_dictionary['black'], 'dataset...', )
        _, _, x_test, labels = load_eurosat()
        num_test = 2700
    elif input_shape == (224, 224, 3) and num_classes == 21:
        dataset = 'ucm'
        print('Using', color_dictionary['red'], 'UCM', color_dictionary['black'], 'dataset...')
        _, _, x_test, labels = load_ucm()
        num_test = 210
    else:
        exit('Invalid model!')

    # Add the input filter name to the dataset
    if input_filter != '':
        dataset += '_' + input_filter

    # Nengo does not like regularization and dropout so we have to create a new model without them
    # - Input layer
    input_layer = tf.keras.Input(shape=input_shape)

    # - If the first layer in the loaded model is an input layer, we skip it
    n = 0 if isinstance(model.layers[0], tf.keras.layers.Conv2D) else 1

    # - First convolutional layer
    config = model.layers[n].get_config()
    x = tf.keras.layers.Conv2D(filters=config['filters'],
                               kernel_size=config['kernel_size'],
                               strides=config['strides'],
                               padding=config['padding'],
                               activation=tf.nn.relu,
                               name=config['name'])(input_layer)

    # - The remaining layers
    for layer in model.layers[n + 1:]:
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Conv2D):
            x = tf.keras.layers.Conv2D(filters=config['filters'],
                                       kernel_size=config['kernel_size'],
                                       strides=config['strides'],
                                       padding=config['padding'],
                                       activation=tf.nn.relu,
                                       name=config['name'])(x)

        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            x = tf.keras.layers.AveragePooling2D(pool_size=config['pool_size'],
                                                 strides=config['strides'],
                                                 padding=config['padding'])(x)

    # - Conclude VGG layers with a global average pooling layer
    global_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

    # - Output layer
    output_layer = tf.keras.layers.Dense(units=num_classes,
                                         use_bias=False,
                                         name=model.layers[-1].get_config()['name'])(global_pool)

    # - Define the  new model
    new_model = tf.keras.Model(input_layer, output_layer)

    # - After the model is defined, we can load the weights
    for layer in new_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            weights = model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)

    # - Compile the new model
    new_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # - Show model's summary
    new_model.summary()

    # Apply preprocessing function and batch
    x_test = x_test.map(rescale_resize(image_size=input_shape[:-1]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(num_test)

    # Apply input filter (x_test must be batched!)
    x_test = x_test.map(input_filter_map(filter_name=input_filter), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Evaluate the model
    loss, acc = new_model.evaluate(x=x_test)
    print('Model accuracy: {:.2f}%.'.format(acc * 100))

    # Convert to a Nengo network
    converter = nengo_dl.Converter(new_model,
                                   scale_firing_rates=scale,
                                   synapse=synapse,
                                   swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()})

    # Input and output objects
    network_input = converter.inputs[input_layer]
    network_output = converter.outputs[output_layer]

    # global_pool probe
    sample_neurons = np.linspace(0, np.prod(global_pool.shape[1:]), N_NEURONS, endpoint=False, dtype=np.int32)
    with converter.net:
        probe = nengo.Probe(converter.layers[global_pool][sample_neurons])
        nengo_dl.configure_settings(stateful=False)

    # Convert the test data from tf.dataset to numpy arrays
    test_data = [(n[0].numpy(), n[1].numpy()) for n in x_test.take(1)]
    x_test = np.array([n[0] for n in test_data])[0]
    y_test = np.array([n[1] for n in test_data])[0]

    # Tile images according to the number of timesteps
    tiled_test_images = np.tile(np.reshape(x_test, (x_test.shape[0], 1, -1)), (1, timesteps, 1))
    test_labels = y_test.reshape((y_test.shape[0], 1, -1))

    # Run simulator
    with nengo_dl.Simulator(converter.net) as sim:
        # Record how much time it takes
        start = time()
        data = sim.predict({network_input: tiled_test_images})
        print('Time to make a prediction with {} timestep(s): {}.'.format(timesteps, timedelta(seconds=time() - start)))

    # Predictions and accuracy
    predictions = np.argmax(data[network_output][:, -1], axis=-1)
    accuracy = (predictions == test_labels[..., 0, 0]).mean()
    print('Test accuracy: {:.2f}% (firing rate scale factor: {}, synapse: {}).'.format(100 * accuracy, scale, synapse))

    # Plot the spikes against the timesteps
    model_name = path_to_model.stem
    path_to_figures = Path('figs/vgg16').joinpath(dataset,
                                                  model_name,
                                                  'scale_{}'.format(scale),
                                                  'synapse_{}'.format(synapse),
                                                  'timesteps_{}'.format(timesteps))
    os.makedirs(path_to_figures, exist_ok=True)

    for i in range(0, N_EXAMPLES, 2):
        plot_spikes(path_to_save=path_to_figures.joinpath('acc_{}_{}.png'.format(accuracy, i // 2)),
                    examples=((255 * x_test).astype('uint8'), y_test),
                    start=i,
                    stop=i + 2,
                    labels=labels,
                    simulator=sim,
                    data=data,
                    probe=probe,
                    network_output=network_output,
                    n_steps=timesteps,
                    scale=scale)


if __name__ == '__main__':
    main()
