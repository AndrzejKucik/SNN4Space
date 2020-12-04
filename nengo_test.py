#!/usr/bin/env python3.6

"""Testing a conversion of a Keras model into a Nengo network, and evaluating its performance on the MNIST dataset.
   Losely based on: https://www.nengo.ai/nengo-dl/examples/keras-to-snn.html"""

# -- Built-in modules -- #
import datetime

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2020-12-04'

# - Assertions to ensure modules compatibility - #
assert nengo.__version__ == '3.0.0', 'Nengo version is {}, and it should be 3.0.0 instead.'.format(nengo.__version__)
assert nengo_dl.__version__ == '3.3.0', 'NengoDL version is {}, and it should be 3.3.0'.format(nengo_dl.__version__)

# - Hyper-parameters - #
# -- Keras model parameters -- #
NUM_CLASSES = 10
EPOCHS = 5

# -- Converter parameters -- #
SWAP_ACTIVATIONS = {tf.nn.relu: nengo.SpikingRectifiedLinear()}
SCALE_FIRING_RATES = 100
SYNAPSE = 0.01

# -- Simulation parameters -- #
N_STEPS = 30

# -- Display parameters -- #
N_NEURONS = 1000
N_EXAMPLES = 3


# Main
def main():
    """The main function."""

    # Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, -1) / 255., np.expand_dims(x_test, -1) / 255.

    # Keras model
    input_layer = tf.keras.Input(shape=x_train.shape[1:])
    conv1 = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   strides=2, activation=tf.nn.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                   use_bias=False)(input_layer)
    conv2 = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=2, activation=tf.nn.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                   use_bias=False)(conv1)
    flatten = tf.keras.layers.Flatten()(conv2)
    dense = tf.keras.layers.Dense(NUM_CLASSES, use_bias=False)(flatten)
    model = tf.keras.Model(input_layer, dense)
    model.summary()
    model.compile(optimizer=tf.optimizers.RMSprop(0.001),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=tf.metrics.SparseCategoricalAccuracy())

    # Prepare logs for tensorboard
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    model.fit(x=x_train, y=y_train, epochs=EPOCHS, callbacks=[tensorboard_callback])

    # Evaluate the model
    _, accuracy = model.evaluate(x=x_test, y=y_test)
    print('Model accuracy: {:.2f}%'.format(100 * accuracy))

    # Convert to a Nengo network
    converter = nengo_dl.Converter(model=model,
                                   allow_fallback=False,
                                   inference_only=True,
                                   scale_firing_rates=SCALE_FIRING_RATES,
                                   synapse=SYNAPSE,
                                   swap_activations=SWAP_ACTIVATIONS)

    # Input and output objects
    network_input = converter.inputs[input_layer]
    network_output = converter.outputs[dense]

    # conv1 probe
    sample_neurons = np.linspace(0, np.prod(conv1.shape[1:]), N_NEURONS, endpoint=False, dtype=np.int32)
    with converter.net:
        probe = nengo.Probe(converter.layers[conv1][sample_neurons])
        nengo_dl.configure_settings(stateful=False)

    # Tile the images
    tiled_test_images = np.tile(np.reshape(x_test, (x_test.shape[0], 1, -1)), (1, N_STEPS, 1))
    test_labels = y_test.reshape((y_test.shape[0], 1, -1))

    # Run simulator
    with nengo_dl.Simulator(converter.net) as sim:
        data = sim.predict({network_input: tiled_test_images})
        predictions = np.argmax(data[network_output][:, -1], axis=-1)
        accuracy = (predictions == test_labels[..., 0, 0]).mean()
        print('Test accuracy: {:.2f}%'.format(100 * accuracy))

        # Plot the results
        plt.figure(figsize=(N_EXAMPLES, 3))
        for i in range(N_EXAMPLES):
            # Input image
            plt.subplot(N_EXAMPLES, 3, 3 * i + 1)
            plt.imshow(x_test[i, :, :, 0], cmap='gray')
            plt.title('Input image')
            plt.axis('off')

            # Sample layer activations
            plt.subplot(N_EXAMPLES, 3, 3 * i + 2)
            scaled_data = data[probe][i] * SCALE_FIRING_RATES
            scaled_data *= 0.001
            plt.plot(scaled_data)
            rates = np.sum(scaled_data, axis=0) / (N_STEPS * sim.dt)
            plt.ylabel('Number of spikes')
            plt.title(f'Neural activities (conv1 mean={rates.mean():.1f} Hz, max={rates.max():.1f} Hz)')

            # Output predictions
            plt.subplot(N_EXAMPLES, 3, 3 * i + 3)
            plt.title('Output predictions')
            plt.plot(tf.nn.softmax(data[network_output][i]))
            plt.legend([str(j) for j in range(10)], loc='upper left')
            plt.xlabel('Timestep')
            plt.ylabel('Probability')

        plt.show()


if __name__ == '__main__':
    main()
