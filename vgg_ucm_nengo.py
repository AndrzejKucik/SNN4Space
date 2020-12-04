#!/usr/bin/env python3.6

"""Converts a Tensorflow model trained on the UC merced dataset into a Nengo spiking neural network and evaluates it."""

# -- Built-in modules -- #
from argparse import ArgumentParser
from time import time
from datetime import timedelta

# -- Third-party modules -- #
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- '
import vgg_ucm
import utils

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2020-12-04'

# - Assertions to ensure modules compatibility - #
assert nengo.__version__ == '3.0.0', 'Nengo version is {}, and it should be 3.0.0 instead.'.format(nengo.__version__)
assert nengo_dl.__version__ == '3.3.0', 'NengoDL version is {}, and it should be 3.3.0'.format(nengo_dl.__version__)

# - Parameters - #
N_NEURONS = 1000
SYNAPSES = [0.001, 0.005, 0.01]
SCALES = [5, 10, 50, 100, 500]
TIMESTEPS = [5, 10, 50, 100, 500]

# - Argument parser - #
parser = ArgumentParser()
parser.add_argument('-md', '--model_path', type=str, default='', help='Path to the model.')
args = vars(parser.parse_args())
PATH_TO_MODEL = args['model_path']

# Main
def main():
    """The main function."""
    # Load data
    (ucm_train, ucm_test, ucm_val), info = vgg_ucm.load_data(show_examples=False)

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=PATH_TO_MODEL)

    # If there is no valid model, a new one will be created and trained
    except OSError:
        print('Invalid model path')
        model = vgg_ucm.create_model()
        vgg_ucm.train_model(model=model, train_data=ucm_train, val_data=ucm_val)

    # Nengo does not like regularization and dropout so we have to create a new model without them
    # - Input layer
    input_layer = tf.keras.Input(shape=vgg_ucm.INPUT_SHAPE)

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
    output_layer = tf.keras.layers.Dense(vgg_ucm.NUM_CLASSES, use_bias=False,
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
    loss, acc = model.evaluate(x=ucm_test.batch(vgg_ucm.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))
    print('Model accuracy: {:.2f}%.'.format(acc * 100, 2))

    # Convert the test data from tf.dataset to numpy arrays
    test_data = [(n[0].numpy(), n[1].numpy()) for n in ucm_test.take(210)]
    x_test = np.array([n[0] for n in test_data])
    y_test = np.array([n[1] for n in test_data])

    # Run the simulations for all the possible parameter combinations
    for synapse in SYNAPSES:
        print('Synapse: ', synapse)

        for scale in SCALES:
            print('Firing rate scale: ', scale)

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

            # Placeholder for model accuracies
            accuracies = []

            for steps in TIMESTEPS:
                # Tile images according to the number of timesteps
                tiled_test_images = np.tile(np.reshape(x_test, (x_test.shape[0], 1, -1)), (1, steps, 1))
                test_labels = y_test.reshape((y_test.shape[0], 1, -1))

                # Run simulator
                with nengo_dl.Simulator(converter.net) as sim:
                    # Record how much time it takes
                    start = time()
                    data = sim.predict({network_input: tiled_test_images})
                    stop = time()
                    print('Time to make a prediction with {} timestep(s): {}.'.format(steps,
                                                                                      timedelta(seconds=stop - start)))

                    # Predictions and accuracy
                    predictions = np.argmax(data[network_output][:, -1], axis=-1)
                    accuracy = (predictions == test_labels[..., 0, 0]).mean()
                    print('Test accuracy: {:.2f}%'.format(100 * accuracy))

                    # Append to the list of accuracies
                    accuracies.append(accuracy)

                # Plot the spikes against the timesteps
                utils.plot_spikes(title='acc_{:.2f}_synapse_{}_scale_{}_steps_{}'.format(100 * accuracy, synapse,
                                                                                         scale, steps),
                                  examples=((255 * x_test[:2]).astype('uint8'), y_test[:2]),
                                  labels=info.features['label'].names,
                                  simulation=sim,
                                  data=data,
                                  probe=probe,
                                  network_output=network_output,
                                  n_steps=steps,
                                  scale=scale)

            # Show how the accuracy increases with the number of steps
            utils.plot_timestep_accuracy(timesteps=TIMESTEPS,
                                         accuracies=accuracies,
                                         scale_firing_rates=scale,
                                         synapse=synapse,
                                         show=False)


if __name__ == '__main__':
    main()
