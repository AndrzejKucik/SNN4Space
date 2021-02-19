#!/usr/bin/env python3.6

"""Estimators of flops, MACs, ACs, and energy consumption of ANN and SNN models during the inference.."""

# -- Built-in modules -- #
import csv
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
from time import time

# -- Third-party modules -- #
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
from tqdm import trange

# -- Proprietary modules -- #
from dataloaders import load_eurosat, load_ucm
from utils import COLOUR_DICTIONARY, input_filter_map, rescale_resize

# -- File info -- #
__author__ = ['Andrzej S. Kucik', 'Gabriele Meoni']
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.2'
__date__ = '2021-02-19'

COLUMN_NAMES = ['date',  # Date
                'model_path', 'input_filter', 'batch_size',  # Model  parameters
                'firing_rate_scale', 'synapse',  # Neuron parameters
                'timesteps', 'simulation_length',  # Timing data
                'ann_accuracy', 'snn_accuracy',  # Accuracies
                'cpu_neuron_energy', 'cpu_synop_energy', 'cpu_total_energy',  # CPU
                'gpu_neuron_energy', 'gpu_synop_energy', 'gpu_total_energy',  # GPU
                'myriad2_neuron_energy', 'myriad2_synop_energy', 'myriad2_total_energy',  # Myriad2
                'loihi_neuron_energy', 'loihi_synop_energy', 'loihi_total_energy',  # Loihi
                'spinnaker_neuron_energy', 'spinnaker_synop_energy', 'spinnaker_total_energy',  # SpiNNaker
                'spinnaker2_neuron_energy', 'spinnaker2_synop_energy', 'spinnaker2_total_energy'  # SpiNNaker2
                ]


def compute_connections(layer):
    """
    Compute the average number of connections for neuron in a layer.

    Parameters
    ----------
    layer : tf.keras.layer
        Keras layer.

    Returns
     -------
    num_input_connections : int
        Number of average number connections per neuron.
    """

    # Input tensor
    input_tensor = layer.input

    # Config of the layer
    config = layer.get_config()

    if isinstance(layer, tf.keras.layers.InputLayer):
        num_input_connections = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        input_channels = (input_tensor.shape[-1] if config['data_format'] == 'channels_last' else input_tensor.shape[1])
        num_input_connections = np.prod(config['kernel_size']) * input_channels

    elif isinstance(layer, tf.keras.layers.Dense):
        num_input_connections = input_tensor.shape[-1]

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        num_input_connections = np.prod(config['pool_size'])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        if config['data_format'] == 'channels_last':
            num_input_connections = np.prod(input_tensor.shape[1:2])
        else:
            num_input_connections = np.prod(input_tensor.shape[-2:-1])

    # We assume that all other layer types are just passing the input
    else:
        num_input_connections = 0

    return num_input_connections


def compute_neurons(layer):
    """
    Calculate the number of neurons in a layer.
    Parameters
    ----------
    layer : tf.keras.layer
        Keras layer.

    Returns
    -------
    neurons : int
        Number of neurons in the layer.
    """

    neurons = 0
    if isinstance(layer, tf.keras.layers.InputLayer):
        neurons = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        neurons = np.prod(layer.output_shape[1:])

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        neurons = np.prod(layer.output_shape[1:])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        neurons = np.prod(layer.output_shape[1:])
    elif isinstance(layer, tf.keras.layers.Dense):
        neurons = layer.output_shape[1]
    else:
        print('\tNothing to do.\n')

    return neurons


def energy_estimate_layer(model,
                          layer_idx: int,
                          spikes_measurements=None,
                          probe_layers=None,
                          dt: float = 0.001,
                          spiking_model=True,
                          device: str = 'cpu',
                          if_neurons: bool = False,
                          verbose: bool = False):
    """
    Estimate the energy spent for synaptic operations and neurons update required for an inference for an Artificial or
    Spiking layer on a target hardware. Energy is estimated by multiplying the number of synaptic operations and neuron
    updates times the energy values for a single synaptic operation and neuron update. Energy values for synaptic
    operation and neuron update are obtained by KerasSpiking: https://www.nengo.ai/keras-spiking/.
    Parameters
    ----------
    model : tf.keras.Model
        Keras model.
    layer_idx : int
        Layer index.
    spikes_measurements: SimulationData (default=None)
        Simulation data produced during the simulation of the Nengo model. It can be None for Artificial Neural
        Networks.
    probe_layers: list (default=None)
        List of probes for the different layers of the Nengo model.
    dt: float (Default=0.001)
        Nengo simulator time resolution
    spiking_model: bool (default=True)
        Flag to indicate if the model is a spiking model or not
    device: str (default='cpu')
        Device name of the target hardware. Supported `cpu` (Intel i7-4960X), `gpu` (Nvidia GTX Titan Black), `arm` (ARM
        Cortex-A), `loihi`, `spinnaker`, `spinnaker2`.
    if_neurons: bool (default=False)
        If `True`, Integrate and Fire (IF) neurons are modelled.  Such neurons can be updated only when an input spike
        is provided (if the hardware supports sporadic activations).
        If `False`, Leaky Integrate and Fire neurons are modelled, which are updated once per timestep. (Note: use this
        even for IF neurons with synaptic filters).
    verbose: bool (default=False)
        If `True`, energy contributions are shown for every single layer.
    Returns
    -------
    synop_energy  : int
        Energy contribution for synaptic operations.
    neuron_energy : int
        Energy contribution for neuron updates.
    """

    # Energy constants for the different operations
    devices = {
        # https://ieeexplore.ieee.org/abstract/document/7054508
        'cpu': dict(spiking=False, energy_per_synop=8.6e-9, energy_per_neuron=8.6e-9),
        'gpu': dict(spiking=False, energy_per_synop=0.3e-9, energy_per_neuron=0.3e-9),
        'arm': dict(spiking=False, energy_per_synop=0.9e-9, energy_per_neuron=0.9e-9),
        'myriad2': dict(spiking=False, energy_per_synop=1.9918386085474344e-10,
                        energy_per_neuron=1.9918386085474344e-10),

        # Value estimated by considering the energy for a MAC operation. Such energy (E_per_mac) is obtained through a
        # maximum-likelihood estimation: E_inf = E_per_mac * N_ops by, E_inf and N_ops values come from our previous
        # work: https://ieeexplore.ieee.org/abstract/document/8644728

        # https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
        'loihi': dict(spiking=True, energy_per_synop=(23.6 + 3.5) * 1e-12, energy_per_neuron=81e-12),

        # https://arxiv.org/abs/1903.08941
        'spinnaker': dict(spiking=True, energy_per_synop=13.3e-9, energy_per_neuron=26e-9),
        'spinnaker2': dict(spiking=True, energy_per_synop=450e-12, energy_per_neuron=2.19e-9),
    }

    energy_dict = devices[device]
    if spiking_model and not energy_dict['spiking']:
        print('Error! Impossible to infer Spiking models on standard hardware!')
        return -1.0, -1.0

    # Energy per synaptic operation
    energy_per_synop = energy_dict['energy_per_synop']

    # Energy per neuron update
    energy_per_neuron = energy_dict['energy_per_neuron']

    # Number of neurons in a layer
    n_neurons = compute_neurons(model.layers[layer_idx])

    # Average number of input connections per neuron
    n_connections_per_neuron = compute_connections(model.layers[layer_idx])
    n_timesteps = 0
    if not spiking_model:
        f_in = 1 / dt
        n_timesteps = 1
    else:
        if layer_idx == 0:
            f_in = 0
        else:
            spikes_in = spikes_measurements[probe_layers[layer_idx - 1]]
            n_timesteps = spikes_in.shape[1]
            f_in = np.sum(spikes_in) / (spikes_in.shape[0] * n_timesteps * spikes_in.shape[2] * dt)

    # synop_energy/inference = energy/op * ops/event * events/s * s/timestep * timesteps/inference
    synop_energy = (energy_per_synop
                    * n_connections_per_neuron
                    * n_neurons
                    * f_in
                    * dt
                    * n_timesteps)

    if if_neurons:  # neurons are update only when an output spike is produced
        # neuron_energy/inference = energy/op * ops/event * events/s * s/timestep * timesteps/inference
        neuron_energy = energy_per_neuron * n_neurons * f_in * dt * n_timesteps
    else:  # neurons are update every timestep (for instance, to implement alpha functions)
        # neuron_energy/inference = energy/op * ops/timestep * timesteps/inference
        neuron_energy = energy_per_neuron * n_neurons * n_timesteps

    if verbose:
        print('--- Layer: ', model.layers[layer_idx].name, ' ---')
        print('\tSynop energy: ', synop_energy, 'J/inference')
        print('\tNeuron energy: ', neuron_energy, 'J/inference')
        print('\tTotal energy: ', neuron_energy + synop_energy, 'J/inference')

    return synop_energy, neuron_energy


def energy_estimate_model(model,
                          spikes_measurements=None,
                          probe_layers=None,
                          dt: float = 0.001,
                          spiking_model=True,
                          device: str = 'cpu',
                          if_neurons: bool = False,
                          verbose: bool = False):
    """Applies energy_estimate_layer to the layers of model iteratively and returns the accumulated results"""

    # Initialize with zeros
    neuron_energy, synop_energy = 0, 0

    # Loop over the layers
    for layer_idx in range(len(model.layers)):
        neuron_energy_layer, synop_energy_layer = energy_estimate_layer(model,
                                                                        layer_idx,
                                                                        spikes_measurements=spikes_measurements,
                                                                        probe_layers=probe_layers,
                                                                        dt=dt,
                                                                        spiking_model=spiking_model,
                                                                        device=device,
                                                                        if_neurons=if_neurons,
                                                                        verbose=False)

        # - Accumulate the result
        neuron_energy += neuron_energy_layer
        synop_energy += synop_energy_layer

    if verbose:
        print(COLOUR_DICTIONARY['orange'], '\t--------- Total energy ---------', COLOUR_DICTIONARY['black'])
        print('\tSynop energy: ', synop_energy, 'J/inference')
        print('\tNeuron energy: ', neuron_energy, 'J/inference')
        print('\tTotal energy:', COLOUR_DICTIONARY['green'], synop_energy + neuron_energy, COLOUR_DICTIONARY['black'],
              'J/inference\n\n')

    return np.array([neuron_energy, synop_energy])


# noinspection PyUnboundLocalVariable
def main():
    """The main function."""

    # - Argument parser - #
    parser = ArgumentParser()
    parser.add_argument('-md',
                        '--model_path',
                        type=str,
                        required=True,
                        help='Path to the model.')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=1,
                        help='Batch size for the SNN testing. Must be a positive integer')
    parser.add_argument('-if',
                        '--input_filter',
                        type=str,
                        default='',
                        help='Type of the input filter (if any).')
    parser.add_argument('-sc',
                        '--scale',
                        type=float,
                        default=1.,
                        help='Post simulation scale value.')
    parser.add_argument('-syn',
                        '--synapse',
                        type=float,
                        default=None,
                        help='Synaptic filter alpha constant.')
    parser.add_argument('-t',
                        '--timesteps',
                        type=int,
                        default=1,
                        help='Number of timesteps for the spiking model simulation.')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        default=False,
                        help='If True, the energy contributions for all the layers is shown.')

    args = vars(parser.parse_args())
    path_to_model = args['model_path']
    batch_size = args['batch_size']
    input_filter = args['input_filter'].lower()
    scale = args['scale']
    synapse = args['synapse']
    timesteps = args['timesteps']
    verbose = args['verbose']

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)
    except OSError:
        exit('Invalid model path!')

    # Input and output shapes
    input_shape = model.input.shape[1:]
    num_classes = model.output.shape[-1]

    # Different datasets
    if input_shape == (64, 64, 3) and num_classes == 10:
        dataset = 'EuroSAT'
        n_test = 2700
        _, _, x_test, labels = load_eurosat()
    elif input_shape == (224, 224, 3) and num_classes == 21:
        dataset = 'UCM'
        _, _, x_test, labels = load_ucm()
        n_test = 210
    else:
        exit('Invalid model!')

    print('Using', COLOUR_DICTIONARY['red'], dataset, COLOUR_DICTIONARY['black'], 'dataset...', )

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
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # - Output layer
    output_layer = tf.keras.layers.Dense(units=num_classes,
                                         use_bias=False,
                                         name=model.layers[-1].get_config()['name'])(x)

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
    if verbose:
        new_model.summary()

    # Test ANN
    # - Preprocess the test data
    # -- Apply preprocessing function and batch
    x_test = x_test.map(rescale_resize(image_size=input_shape[:-1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # -- Apply input filter (x_test must be batched!)
    x_test = x_test.batch(batch_size=n_test)
    x_test = x_test.map(input_filter_map(filter_name=input_filter), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Evaluate the ANN model
    _, ann_acc = new_model.evaluate(x=x_test, verbose=verbose)

    # Test SNN
    # - Convert to a Nengo network
    print(COLOUR_DICTIONARY['green'], 'Converting Keras model to Nengo network...', COLOUR_DICTIONARY['black'])
    converter = nengo_dl.Converter(new_model,
                                   scale_firing_rates=scale,
                                   synapse=synapse,
                                   swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()})

    # - Input and output objects
    network_input = converter.inputs[input_layer]
    network_output = converter.outputs[output_layer]

    # - Probe layers
    probe_layers = []
    with converter.net:
        for layer in new_model.layers:
            probe_layers.append(nengo.Probe(converter.layers[layer]))
        nengo_dl.configure_settings(stateful=False)

    # - Convert the test data from tf.dataset to numpy arrays
    test_data = [(n[0].numpy(), n[1].numpy()) for n in x_test.take(1)]
    x_test = np.array([n[0] for n in test_data]).squeeze(0)
    y_test = np.array([n[1] for n in test_data]).squeeze(0)

    # - Tile images according to the number of timesteps
    tiled_test_images = np.tile(np.reshape(x_test, (x_test.shape[0], 1, -1)), (1, timesteps, 1))
    test_labels = y_test.reshape((y_test.shape[0], 1, -1))

    # - Placeholders
    energy_estimates = {'loihi': np.zeros((2,)), 'spinnaker': np.zeros((2,)), 'spinnaker2': np.zeros((2,))}
    snn_acc = 0.

    # - Number of test steps
    n_test_steps = int(n_test / batch_size)

    # - Run the simulations
    print(COLOUR_DICTIONARY['blue'], 'Start simulations...', COLOUR_DICTIONARY['black'])
    # -- Record how much time it takes
    start = time()
    with nengo_dl.Simulator(converter.net, progress_bar=False) as sim:
        for i in trange(n_test_steps):
            data = sim.predict({network_input: tiled_test_images[i * batch_size:(i + 1) * batch_size]}, stateful=False)

            # -- Predictions and accuracy
            predictions = np.argmax(data[network_output][:, -1], axis=-1)
            snn_acc += (predictions == test_labels[i * batch_size:(i + 1) * batch_size, 0, 0]).mean() / n_test_steps

            # -- SNN energy estimation
            for device_snn in ['loihi', 'spinnaker', 'spinnaker2']:
                energy_estimates[device_snn] += energy_estimate_model(model,
                                                                      spikes_measurements=data,
                                                                      probe_layers=probe_layers,
                                                                      dt=sim.dt,
                                                                      spiking_model=True,
                                                                      device=device_snn,
                                                                      verbose=False) / n_test_steps

    # -- Stop the timer
    simulation_len = timedelta(seconds=time() - start)

    # -- ANN energy estimation (only a single run required)
    for device_ann in ['cpu', 'gpu', 'myriad2']:
        energy_estimates[device_ann] = energy_estimate_model(model,
                                                             spikes_measurements=data,
                                                             probe_layers=probe_layers,
                                                             dt=sim.dt,
                                                             spiking_model=False,
                                                             device=device_ann,
                                                             verbose=False)

    # Write the results to a CSV file
    # - Create the file if it does not exist
    if not os.path.isfile('estimates.csv'):
        with open('estimates.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(COLUMN_NAMES)

    # - Append the results
    with open('estimates.csv', mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow([
            # Date
            datetime.now().strftime('%d/%m/%Y %H:%M:%S'),

            # Model parameters
            path_to_model, input_filter, batch_size,

            # Neuron parameters
            scale, synapse,

            # Timing data
            timesteps, simulation_len,

            # Accuracies
            ann_acc, snn_acc,

            # CPU
            energy_estimates['cpu'][0],
            energy_estimates['cpu'][1],
            np.sum(energy_estimates['cpu']),

            # GPU
            energy_estimates['gpu'][0],
            energy_estimates['gpu'][1],
            np.sum(energy_estimates['gpu']),

            # Myriad2
            energy_estimates['myriad2'][0],
            energy_estimates['myriad2'][1],
            np.sum(energy_estimates['myriad2']),

            # Loihi
            energy_estimates['loihi'][0],
            energy_estimates['loihi'][1],
            np.sum(energy_estimates['loihi']),

            # SpiNNaker
            energy_estimates['spinnaker'][0],
            energy_estimates['spinnaker'][1],
            np.sum(energy_estimates['spinnaker']),

            # SpiNNaker2
            energy_estimates['spinnaker2'][0],
            energy_estimates['spinnaker2'][1],
            np.sum(energy_estimates['spinnaker2'])
        ])

    # Print results
    if verbose:
        # - Time and batch info
        print('\n\nTime to make SNN predictions with',
              COLOUR_DICTIONARY['blue'], n_test, COLOUR_DICTIONARY['black'],
              'examples and with',
              COLOUR_DICTIONARY['blue'], timesteps, COLOUR_DICTIONARY['black'],
              'timestep(s): ',
              COLOUR_DICTIONARY['blue'], simulation_len, COLOUR_DICTIONARY['black'], '.\n')

        # - Accuracies
        print(COLOUR_DICTIONARY['purple'], 'ANN test accuracy: {:.2f}%.'.format(ann_acc * 100),
              COLOUR_DICTIONARY['black'])
        print(COLOUR_DICTIONARY['purple'], 'SNN test accuracy: {:.2f}%'.format(100 * snn_acc),
              COLOUR_DICTIONARY['black'],
              ' (firing rate scale factor: {}, synapse: {}).\n'.format(scale, synapse))

        # -- Energy estimates
        for device in ['cpu', 'gpu', 'myriad2', 'loihi', 'spinnaker', 'spinnaker2']:
            if device == 'cpu':
                print(COLOUR_DICTIONARY['cyan'], '--------- ANN model ---------', COLOUR_DICTIONARY['black'])
            elif device == 'loihi':
                print(COLOUR_DICTIONARY['cyan'], '--------- SNN model ---------', COLOUR_DICTIONARY['black'])

            print('\tHardware: ', COLOUR_DICTIONARY['red'], device, COLOUR_DICTIONARY['black'])
            print('\tSynop energy: ', energy_estimates[device][0], 'J/inference')
            print('\tNeuron energy: ', energy_estimates[device][1], 'J/inference')
            print('\tTotal energy:', COLOUR_DICTIONARY['green'], np.sum(energy_estimates[device]),
                  COLOUR_DICTIONARY['black'], 'J/inference\n\n')


# Main function
if __name__ == '__main__':
    main()
