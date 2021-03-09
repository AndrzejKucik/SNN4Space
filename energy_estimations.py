#!/usr/bin/env python3.6

"""Estimators of flops, MACs, ACs, and energy consumption of ANN and SNN models during the inference.."""

# -- Built-in modules -- #
import csv
import os
from argparse import ArgumentParser
from datetime import datetime

# -- Third-party modules -- #
import keras_spiking
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from create_models import create_spiking_vgg16_model
from dataloaders import load_eurosat, load_ucm
from utils import add_temporal_dim, COLOUR_DICTIONARY, input_filter_map, rescale_resize

# -- File info -- #
__author__ = ['Andrzej S. Kucik', 'Gabriele Meoni']
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.3.0'
__date__ = '2021-03-09'

# Entry names for the CSV file
COLUMN_NAMES = ['date',  # Date
                'weights_path', 'dataset', 'batch_size',  # Model  parameters
                'timesteps', 'dt',  # Timing data
                'ann_cpu_synop_energy', 'ann_cpu_neuron_energy', 'ann_cpu_total_energy',
                'ann_gpu_synop_energy', 'ann_gpu_neuron_energy', 'ann_gpu_total_energy',
                'ann_myriad2_synop_energy', 'ann_myriad2_neuron_energy', 'ann_myriad2_total_energy',
                'ann_loihi_synop_energy', 'ann_loihi_neuron_energy', 'ann_loihi_total_energy',
                'ann_spinnaker_synop_energy', 'ann_spinnaker_neuron_energy', 'ann_spinnaker_total_energy',
                'ann_spinnaker2_synop_energy', 'ann_spinnaker2_neuron_energy', 'ann_spinnaker2_total_energy',
                'snn_loihi_synop_energy', 'snn_loihi_neuron_energy', 'snn_loihi_total_energy',
                'snn_spinnaker_synop_energy', 'snn_spinnaker_neuron_energy', 'snn_spinnaker_total_energy',
                'snn_spinnaker2_synop_energy', 'snn_spinnaker2_neuron_energy', 'snn_spinnaker2_total_energy'
                ]

# Device parameters per energy estimation
DEVICES = {
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

# - Argument parser - #
parser = ArgumentParser()
# -- Model parameters
parser.add_argument('-wp',
                    '--weights_path',
                    type=str,
                    default='',
                    help='Path to the model weights.')
# -- Dataset
parser.add_argument('-ds',
                    '--dataset',
                    type=str,
                    default='eurosat',
                    help='Dataset. Either `eurosat` or `ucm`. One can also add `prewitt`, `sobel`, `mask` or `sq`.')
# -- Simulation parameters
parser.add_argument('-bs',
                    '--batch_size',
                    type=int,
                    default=1,
                    help='Batch size for the SNN testing. Must be a positive integer')
parser.add_argument('-t',
                    '--timesteps',
                    type=int,
                    default=1,
                    help='The length of the simulation. Must be a positive integer')
parser.add_argument('-dt',
                    '--dt',
                    type=float,
                    default=.001,
                    help='Temporal resolution of the simulation.')
# -- Verbosity
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    default=False,
                    help='If True, the energy contributions for all the layers is shown.')

# -- Parse arguments
args = vars(parser.parse_args())
WEIGHTS_PATH = args['weights_path']
DATASET = args['dataset'].lower()
BATCH_SIZE = args['batch_size']
TIMESTEPS = args['timesteps']
DT = args['dt']
VERBOSE = args['verbose']

# Fix the dataset parameters
if 'eurosat' in DATASET:
    INPUT_SHAPE = (64, 64, 3)
    NUM_CLASSES = 10
    BUFFER_SIZE = 21600
    print('Using', COLOUR_DICTIONARY['red'], 'EuroSAT', COLOUR_DICTIONARY['black'], 'dataset...', )
elif 'ucm' in DATASET:
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 21
    BUFFER_SIZE = 1680
    print('Using', COLOUR_DICTIONARY['red'], 'UC Merced', COLOUR_DICTIONARY['black'], 'dataset...', )
else:
    exit('Invalid dataset!')


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

    if isinstance(layer, tf.keras.layers.Conv2D):
        input_channels = (input_tensor.shape[-1] if config['data_format'] == 'channels_last' else input_tensor.shape[1])
        num_input_connections = np.prod(config['kernel_size']) * input_channels

    elif isinstance(layer, tf.keras.layers.Dense):
        num_input_connections = input_tensor.shape[-1]

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
        Keras layer object.
    Returns
    -------
    neurons : int
        Number of neurons in the layer.
    """

    if isinstance(layer, tf.keras.layers.Conv2D):
        neurons = np.prod(layer.output_shape[2:])
    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        neurons = np.prod(layer.output_shape[2:])
    elif isinstance(layer, tf.keras.layers.Dense):
        neurons = layer.output_shape[1]
    else:
        neurons = 0

    return neurons


def extract_model_info(model):
    """
    Providing a list of number of input connections per neuron, number of neurons for Conv2D, Dense, and
    GlobalAveragePool2D layers, and providing the indices of the layers for which the activations shall be extracted,
    including LowPass filters, Dense and GlobalAveragePooling activations.

    Parameters
    ----------
    model : tf.keras.Model
        Keras Model object.

    Returns
    -------
    number_of_connection_per_neuron_list : list
        List of the number of input connection per neuron for the different layers.
    number_of_neurons_list : list
        List of the number of neurons for the different layers.
    activations_to_track_index_list : list
        List containing the indices of the layers for which activations shall be extracted.
    """

    # Placeholders
    number_of_connection_per_neuron_list = []
    number_of_neurons_list = []
    activations_to_track_index_list = []

    for i, layer in enumerate(model.layers):
        # Get the indices of the layers sending out spikes
        if (isinstance(layer, tf.keras.layers.Reshape)
                or isinstance(layer, keras_spiking.layers.Lowpass)
                or isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)):
            activations_to_track_index_list.append(i)

        # Get the number of neurons and connections for the layers receiving spikes
        if (isinstance(layer, tf.keras.layers.Conv2D)
                or isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
                or isinstance(layer, tf.keras.layers.Dense)):
            number_of_connection_per_neuron_list.append(compute_connections(layer))
            number_of_neurons_list.append(compute_neurons(layer))

    # Assert that we have all the data we need
    assert (len(number_of_connection_per_neuron_list) == len(number_of_neurons_list)
            == len(activations_to_track_index_list))

    return number_of_connection_per_neuron_list, number_of_neurons_list, activations_to_track_index_list


def energy_estimation(model,
                      x_test=None,
                      spiking_model: bool = True,
                      device_list: list = None,
                      n_timesteps: int = TIMESTEPS,
                      dt: float = DT,
                      verbose: bool = False):
    """
    Estimate the energy spent for synaptic operations and neurons update required for an inference for an Artificial or
    Spiking layer on a target hardware list. Energy is estimated by multiplying the number of synaptic operations and
    neuron updates times the energy values for a single synaptic operation and neuron update. The number of synaptic
    operations is obtained by performing the inference on a test_dataset for a spiking model. Energy values for synaptic
    operation and neuron update are obtained by KerasSpiking: https://www.nengo.ai/keras-spiking/. Energy values for
    Myriad 2 devices is obtained by mean square error interpolation of the values provided in: Benelli, Gionata,
    Gabriele Meoni, and Luca Fanucci. "A low power keyword spotting algorithm for memory constrained embedded systems."
    2018 IFIP/IEEE International Conference on Very Large Scale Integration (VLSI-SoC). IEEE, 2018.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model object
    x_test: tf.raw_ops.ParallelMapDataset (default=None)
        Simulation dataset to extract number of synaptic operations for a spiking model. It can be None for Artificial
        Neural Networks.
    spiking_model: bool (default=True)
        Flag to indicate if the model is a spiking model or not.
    device_list: list (default=['loihi'])
        List containing the name of the target hardware devices. Supported `cpu` (Intel i7-4960X), `gpu` (Nvidia GTX
        Titan Black), `arm` (ARM Cortex-A), `loihi`, `spinnaker`, `spinnaker2`, `myriad2`.
    n_timesteps: int (default=10)
        Number of simulation timesteps. For artificial models, n_timesteps is ignored and 1 is used.
    dt: float (Default=0.001)
        Nengo simulator time resolution.

    verbose: bool (default=False)
        If `True`, energy contributions are shown for every single layer and additional log info is provided.

    Returns
    -------
    synop_energy_dict : dict
        Dictionary including the energy contribution for synaptic operations for each target device.
    neuron_energy_dict : dict
         Dictionary including the energy contribution for neuron updates for each target device.
    total_energy_dict : dict
        Dictionary including the total energy per device.
    """

    if device_list is None:
        device_list = list(DEVICES.keys())

    # Extract the info
    print('Extracting model info...')
    [number_of_connection_per_neuron_list, number_of_neurons_list,
     activations_to_track_index_list] = extract_model_info(model)

    # Energy estimation
    # - Initialize it with the total number of neurons
    neuron_energy = np.sum(number_of_neurons_list)

    # - Spiking model
    if spiking_model:
        print('Found a spiking model. Extracting intermediate activations...')

        # -- In spiking model the neuron energy must be multiplied by the number of timesteps
        neuron_energy *= n_timesteps

        # -- Collect the mean activations across the layers of interest
        mean_activations = [tf.reduce_mean(tf.abs(model.layers[n].output)) for n in activations_to_track_index_list]

        # -- In each layer we initialize the synaptic energy as the number of neurons multiplied by the number of
        # -- connections that each neuron in that layer has, times the average mean input activations
        synop_energy = tf.add_n([number_of_connection_per_neuron_list[n] * number_of_neurons_list[n]
                                 * mean_activations[n] for n in range(len(mean_activations))])

        # -- Define a new model to calculate the the base synaptic energy, given the input
        new_model = tf.keras.Model(model.input, synop_energy)
        new_model.compile()

        # -- Overwrite synop_energy with new_model predictions
        synop_energy = new_model.predict(x_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
        # -- Get the mean across the training batches
        synop_energy = np.mean(synop_energy)
        # -- Then it is multiplied by the number of timesteps and the temporal resolution
        synop_energy *= n_timesteps * dt

    # - Non-spiking model
    else:
        print('Found a non-spiking model.')
        synop_energy = np.dot(np.array(number_of_connection_per_neuron_list), number_of_connection_per_neuron_list)

    # - Placeholders for energy readings
    synop_energy_dict = {}
    neuron_energy_dict = {}
    total_energy_dict = {}

    # - Loop over the devices
    for device in device_list:
        energy_dict = DEVICES[device]
        if spiking_model and not energy_dict['spiking']:
            print(COLOUR_DICTIONARY['red'], 'Error!', COLOUR_DICTIONARY['purple'],
                  'Impossible to infer spiking models on standard hardware!', COLOUR_DICTIONARY['black'])
            break

        # -- Multiply the energy units by the energy consumption specific for a device
        synop_energy_dict[device] = synop_energy * energy_dict['energy_per_synop']
        neuron_energy_dict[device] = neuron_energy * energy_dict['energy_per_neuron']
        total_energy_dict[device] = synop_energy_dict[device] + neuron_energy_dict[device]

        # -- Print out the results if necessary
        if verbose:
            print('Estimating energy on ', COLOUR_DICTIONARY['red'], device, COLOUR_DICTIONARY['black'])
            print(COLOUR_DICTIONARY['red'], 'Global model energy', COLOUR_DICTIONARY['black'])
            print(COLOUR_DICTIONARY['orange'], '\t--------- Total energy ---------', COLOUR_DICTIONARY['black'])
            print('\tSynop layer energy: ', synop_energy_dict[device], 'J/inference')
            print('\tNeuron layer energy: ', neuron_energy_dict[device], 'J/inference')
            print('\tTotal layer energy:', COLOUR_DICTIONARY['green'], total_energy_dict[device],
                  COLOUR_DICTIONARY['black'], 'J/inference\n\n')

    return synop_energy_dict, neuron_energy_dict, total_energy_dict


def main():
    """The main function."""

    # Model
    # - Create a model, initializing it with VGG16 weights
    model = create_spiking_vgg16_model(model_path='',
                                       input_shape=INPUT_SHAPE,
                                       dt=DT,
                                       num_classes=NUM_CLASSES)

    # - Load weights trained in a spiking aware training
    try:
        model.load_weights(filepath=WEIGHTS_PATH)
    except tf.errors.NotFoundError:
        print(COLOUR_DICTIONARY['red'], 'Failed to load the weights!',
              COLOUR_DICTIONARY['purple'], 'Proceeding with VGG16 weights.',
              COLOUR_DICTIONARY['black'])

    # - Print out model's summary
    if VERBOSE:
        model.summary()

    # - Energy usage
    # -- ANN
    ann_synop_energy_dict, ann_neuron_energy_dict, ann_total_energy_dict = energy_estimation(model,
                                                                                             spiking_model=False,
                                                                                             verbose=VERBOSE)

    # -- SNN
    # --- Load data
    x_test = load_eurosat()[2] if 'eurosat' in DATASET else load_ucm()[2]
    x_test = x_test.map(rescale_resize(image_size=INPUT_SHAPE[:-1]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ---- Add the temporal dimension
    x_test = x_test.map(add_temporal_dim(timesteps=TIMESTEPS), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(input_filter_map(filter_name=DATASET), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ---- Discard the labels to conserve the energy and have no inconsistencies in the synaptic energy estimation model
    x_test = x_test.map(lambda x, y: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.batch(batch_size=BATCH_SIZE)
    x_test = x_test.prefetch(tf.data.experimental.AUTOTUNE)

    # --- Get the energy usage
    snn_synop_energy_dict, snn_neuron_energy_dict, snn_total_energy_dict = energy_estimation(model,
                                                                                             x_test=x_test,
                                                                                             spiking_model=True,
                                                                                             device_list=['loihi',
                                                                                                          'spinnaker',
                                                                                                          'spinnaker2'],
                                                                                             n_timesteps=TIMESTEPS,
                                                                                             dt=DT,
                                                                                             verbose=VERBOSE)

    # Write the results to a CSV file
    # - Create the file if it does not exist
    if not os.path.isfile('estimates.csv'):
        with open('estimates.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(COLUMN_NAMES)

    # - Append the results
    with open('estimates.csv', mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        row = [
            # Date
            datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            # Model parameters
            WEIGHTS_PATH, DATASET, BATCH_SIZE,
            # Timing data
            TIMESTEPS, DT
        ]

        # -- ANN
        for device in ['cpu', 'gpu', 'myriad2', 'loihi', 'spinnaker', 'spinnaker2']:
            row += [ann_synop_energy_dict[device], ann_neuron_energy_dict[device], ann_total_energy_dict[device]]

        # -- SNN
        for device in ['loihi', 'spinnaker', 'spinnaker2']:
            row += [snn_synop_energy_dict[device], snn_neuron_energy_dict[device], snn_total_energy_dict[device]]

        csv_writer.writerow(row)


# Main function
if __name__ == '__main__':
    main()
