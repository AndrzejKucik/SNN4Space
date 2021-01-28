#!/usr/bin/env python3.6

"""
Converts a Tensorflow model trained on the EuroSAT or UC Merced dataset into a Nengo spiking neural network and
evaluates it.
"""

# -- Built-in modules -- #
from argparse import ArgumentParser
from datetime import timedelta
import os
from pathlib import Path
from time import time

# -- Third-party modules -- #
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- '
from dataloaders import load_eurosat, load_ucm
import utils


# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.0'
__date__ = '2021-01-28'

# - Assertions to ensure modules compatibility - #
assert nengo.__version__ == '3.0.0', 'Nengo version is {}, and it should be 3.0.0 instead.'.format(nengo.__version__)
assert nengo_dl.__version__ == '3.3.0', 'NengoDL version is {}, and it should be 3.3.0'.format(nengo_dl.__version__)

# - Parameters - #
N_NEURONS = 1000
N_EXAMPLES = 10


def main():
    """The main function."""
    # - Argument parser - #
    parser = ArgumentParser()
    parser.add_argument('-md', '--model_path', type=str, default='', required=True, help='Path to the model.')
    parser.add_argument('-sc', '--firing_rate_scale', type=float, default=1, help='scale factor for the firing rate.')
    parser.add_argument('-syn', '--synapse', type=float, default=None, help='Value of the synapse.')
    parser.add_argument('-t', '--timesteps', type=int, default=1, help='Simulation timesteps.')

    args = vars(parser.parse_args())
    path_to_model = Path(args['model_path'])
    scale = args['firing_rate_scale']
    synapse = args['synapse']
    timesteps = args['timesteps']

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)

    # If there is no valid model, a new one will be created and trained
    except OSError:
        exit('Invalid model path!')

    # Paramters
    input_shape = model.input.shape[1:]
    num_classes = model.output.shape[-1]
    if input_shape == (64, 64, 3) and num_classes == 10:
        dataset = 'eurosat'
    elif input_shape == (224, 224, 3) and num_classes == 21:
        dataset = 'ucm'
    else:
        exit("Invalid model!")

        # Preprocessing function

    def preprocess(image, label):
        """Rescales and resizes the input images."""

        return utils.rescale_resize(image, input_shape[:-1]), label

    # Nengo does not like regularization and dropout so we have to create a new model without them
    # - Input layer
    input_layer = tf.keras.Input(shape=input_shape)

    # - First convolutional layer
    config = model.layers[1].get_config()
    filters, kernel_size, name = config['filters'], config['kernel_size'], config['name']
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               activation=tf.nn.relu,
                               name=name)(input_layer)

    # - The remaining layers
    for layer in model.layers[2:]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            config = layer.get_config()
            filters, kernel_size, name = config['filters'], config['kernel_size'], config['name']
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name=name)(x)
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    # - Conclude VGG layers with a global average pooling layer
    global_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

    # - Output layer
    output_layer = tf.keras.layers.Dense(num_classes, use_bias=False,
                                         name=model.layers[-1].get_config()['name'])(global_pool)

    # - Define the  new model
    new_model = tf.keras.Model(input_layer, output_layer)

    # - After the model is defined, we can load the weights
    for layer in new_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            weights = model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)

    # Show model's summary
    model.summary()

    # Load data
    if dataset == 'eurosat':
        _, _, x_test, labels = load_eurosat()
        num_test = 2700
    else:  # dataset == 'ucm`
        _, _, x_test, labels = load_ucm()
        num_test = 210

    # Apply preprocessing function
    x_test = x_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Evaluate the model
    loss, acc = model.evaluate(x=x_test.batch(num_test))
    print('Model accuracy: {:.2f}%.'.format(acc * 100))

    # Convert to a Nengo network
    converter = nengo_dl.Converter(new_model,
                                   scale_firing_rates=scale,
                                   synapse=synapse,
                                   swap_activations={tf.nn.relu: nengo.LIF()})

    # Input and output objects
    network_input = converter.inputs[input_layer]
    network_output = converter.outputs[output_layer]

    # global_pool probe
    sample_neurons = np.linspace(0, np.prod(global_pool.shape[1:]), N_NEURONS, endpoint=False, dtype=np.int32)
    with converter.net:
        probe = nengo.Probe(converter.layers[global_pool][sample_neurons])
        nengo_dl.configure_settings(stateful=False)

    # Convert the test data from tf.dataset to numpy arrays
    test_data = [(n[0].numpy(), n[1].numpy()) for n in x_test.take(num_test)]
    x_test = np.array([n[0] for n in test_data])
    y_test = np.array([n[1] for n in test_data])

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
        utils.plot_spikes(path_to_save=path_to_figures.joinpath('acc_{}_{}.png'.format(accuracy, i // 2)),
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
