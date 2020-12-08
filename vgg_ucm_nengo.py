#!/usr/bin/env python3.6

"""Converts a Tensorflow model trained on the UC merced dataset into a Nengo spiking neural network and evaluates it."""

# -- Built-in modules -- #
from argparse import ArgumentParser
from datetime import timedelta
import os
from time import time

# -- Third-party modules -- #
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# -- Proprietary modules -- '
import utils

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.1'
__date__ = '2020-12-08'

# - Assertions to ensure modules compatibility - #
assert nengo.__version__ == '3.0.0', 'Nengo version is {}, and it should be 3.0.0 instead.'.format(nengo.__version__)
assert nengo_dl.__version__ == '3.3.0', 'NengoDL version is {}, and it should be 3.3.0'.format(nengo_dl.__version__)

# - Parameters - #
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 21
N_NEURONS = 1000


# Preprocessing functions
def preprocess(image, label):
    """Rescales and resizes the input images."""

    return utils.rescale_resize(image, INPUT_SHAPE[:-1]), label


def main():
    """The main function."""
    # - Argument parser - #
    parser = ArgumentParser()
    parser.add_argument('-md', '--model_path', type=str, default='', help='Path to the model.')
    parser.add_argument('-sc', '--firing_rate_scale', type=float, default=1, help='scale factor for the firing rate.')
    parser.add_argument('-syn', '--synapse', type=float, default=None, help='Value of the synapse.')
    parser.add_argument('-t', '--timesteps', type=int, default=1, help='Simulation timesteps.')

    args = vars(parser.parse_args())
    path_to_model = args['model_path']
    scale = args['firing_rate_scale']
    synapse = args['synapse']
    timesteps = args['timesteps']

    # Load data
    (_, _, ucm_test), info = tfds.load('uc_merced',
                                       split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                       with_info=True,
                                       as_supervised=True)

    # Number of test examples
    test_size = int(info.splits['train'].num_examples * .1)

    # Apply preprocessing function
    ucm_test = ucm_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)

    # If there is no valid model, a new one will be created and trained
    except OSError:
        exit('Invalid model path!')

    # Nengo does not like regularization and dropout so we have to create a new model without them
    # - Input layer
    input_layer = tf.keras.Input(shape=INPUT_SHAPE)

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
    output_layer = tf.keras.layers.Dense(NUM_CLASSES, use_bias=False,
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

    # Evaluate the model
    loss, acc = model.evaluate(x=ucm_test.batch(test_size))
    print('Model accuracy: {:.2f}%.'.format(acc * 100))

    # Convert to a Nengo network
    converter = nengo_dl.Converter(new_model,
                                   scale_firing_rates=scale,
                                   synapse=synapse,
                                   swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()})

    # Input and output objects
    network_input = converter.inputs[input_layer]
    network_output = converter.outputs[output_layer]

    # global_ppol probe
    sample_neurons = np.linspace(0, np.prod(global_pool.shape[1:]), N_NEURONS, endpoint=False, dtype=np.int32)
    with converter.net:
        probe = nengo.Probe(converter.layers[global_pool][sample_neurons])
        nengo_dl.configure_settings(stateful=False)

    # Convert the test data from tf.dataset to numpy arrays
    test_data = [(n[0].numpy(), n[1].numpy()) for n in ucm_test.take(test_size)]
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
    print('Test accuracy: {:.2f}%'.format(100 * accuracy))

    # Plot the spikes against the timesteps
    model_name = os.path.split(path_to_model)[1][:-3]
    path_to_figures = 'figs/' + model_name + '/scale_{}/synapse_{}/timesteps_{}'.format(scale, synapse, timesteps)

    try:
        os.makedirs(path_to_figures)
    except FileExistsError:
        pass

    for i in range(0, 10, 2):
        print(path_to_figures + '/acc_{}_{}.png'.format(accuracy, i))
        utils.plot_spikes(path_to_save=path_to_figures + '/acc_{}_{}.png'.format(accuracy, i),
                          examples=((255 * x_test).astype('uint8'), y_test),
                          start=i,
                          stop=i+2,
                          labels=info.features['label'].names,
                          simulator=sim,
                          data=data,
                          probe=probe,
                          network_output=network_output,
                          n_steps=timesteps,
                          scale=scale)


if __name__ == '__main__':
    main()
