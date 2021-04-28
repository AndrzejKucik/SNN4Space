#!/usr/bin/env python3.6

"""Auxiliary functions used to create the plots used in the accompanying article and the GitHub repository."""

# -- Built-in modules -- #
import csv
import os
from pathlib import Path

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

# -- Proprietary modules -- #
from dataloaders import load_eurosat

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.0'
__date__ = '2021-04-28'


def plot_sample_eurosat_images():
    """Function plotting sample EuroSAT RGB images before and after applying the Prewitt transform."""

    info = ['Annual\nCrop', 'Forest\n', 'Herbaceous\nVegetation', 'Highway\n', 'Industrial\n',
            'Pasture\n', 'Permanent\nCrop', 'Residential\n', 'River\n', 'Sea\nLake']
    # Load data
    x_test, _ = load_eurosat()[2:]

    # Extract images
    used_labels = []
    examples = []
    for (image, label) in x_test.take(100):
        if label in used_labels:
            pass
        else:
            examples.append(image)
            used_labels.append(label)

    # Apply Prewitt transform and tile
    examples = tf.stack(examples)
    images = tfio.experimental.filter.prewitt(examples / 255.) / tf.sqrt(10.)
    images = images * tf.cast(images >= 2 / 255., tf.float32) * 255.
    pad = tf.ones((10, 2, 64, 3)) * 255.
    examples = tf.concat([examples, pad, images], axis=1)

    # Plot
    plt.figure(constrained_layout=True)
    for i in range(9):
        plt.subplot(1, 10, i + 1)
        plt.imshow(examples[i].numpy().astype('uint8'))
        plt.title(info[used_labels[i].numpy()], fontsize=33, y=-.15)
        plt.axis('off')

    # Display the plot
    plt.show()


def plot_energy(estimates_path: str):
    """
    Function plotting the energy requirements of particular hardware platforms.

    Parameters
    ----------
    estimates_path : str
        Path to CSV file with the energy estimates.
    """

    # Read the CSV file
    with open(estimates_path, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        configs = []
        configs_p = []

        for i, row in enumerate(csv_reader):
            config = dict(row)
            model_config = Path(config['weights_path']).stem.split(sep='_')
            config['acc'] = float(model_config[-1])
            if 'prewitt' in str(row):
                configs_p.append(config)
            else:
                configs.append(config)

    # Initialise the plot
    fig, ax = plt.subplots(constrained_layout=True)
    dot_scale = 3000
    fsize = 33

    # Plot total energy
    plt.scatter(x=[config['acc'] for config in configs],
                y=[float(config['snn_spinnaker_total_energy']) for config in configs],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs],
                c='r',
                label='SpiNNaker SNN total energy')

    plt.scatter(x=[config['acc'] for config in configs_p],
                y=[float(config['snn_spinnaker_total_energy']) for config in configs_p],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs_p],
                c='m',
                label='SpiNNaker SNN total energy (Prewitt)')

    plt.scatter(x=[config['acc'] for config in configs],
                y=[float(config['snn_spinnaker2_total_energy']) for config in configs],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs],
                c='b',
                label='SpiNNaker 2 SNN total energy')

    plt.scatter(x=[config['acc'] for config in configs_p],
                y=[float(config['snn_spinnaker2_total_energy']) for config in configs_p],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs_p],
                c='c',
                label='SpiNNaker 2 SNN total energy (Prewitt)')

    plt.scatter(x=[config['acc'] for config in configs],
                y=[float(config['snn_loihi_total_energy']) for config in configs],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs],
                c='g',
                label='Loihi SNN total energy')

    plt.scatter(x=[config['acc'] for config in configs_p],
                y=[float(config['snn_loihi_total_energy']) for config in configs_p],
                s=[dot_scale * float(config['timesteps']) * float(config['dt']) for config in configs_p],
                c='y',
                label='Loihi SNN total energy (Prewitt)')

    # Plot significant lines
    plt.plot([min([config['acc'] for config in configs]) - 1, 96.07],
             [float(configs[0]['ann_cpu_total_energy']), float(configs[0]['ann_cpu_total_energy'])],
             'k-')
    plt.plot([min([config['acc'] for config in configs]) - 1, 96.07],
             [3 * float(configs[0]['ann_gpu_total_energy']), 3 * float(configs[0]['ann_gpu_total_energy'])],
             'k--')
    plt.plot([min([config['acc'] for config in configs]) - 1, 96.07],
             [float(configs[0]['ann_gpu_total_energy']), float(configs[0]['ann_gpu_total_energy'])],
             'k-.')
    plt.plot([min([config['acc'] for config in configs]) - 1, 96.07],
             [float(configs[0]['ann_loihi_total_energy']), float(configs[0]['ann_loihi_total_energy'])],
             'k:')

    # Plot best ANN performance
    plt.axvline(95.07, ls='-', c='k')
    plt.axvline(90.19, ls='--', c='k')
    plt.plot()

    plt.xlabel('Accuracy (%)', fontsize=fsize)
    plt.xticks([float(round(config['acc'])) for config in configs + configs_p] + [90.19, 95.07],
               [str(round(config['acc'])) for config in configs + configs_p] + ['90.19', '95.07'],
               fontsize=fsize)
    secax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    secax.set_ticks([90.19, 95.07])
    secax.set_xticklabels(['Best ANN+Prewitt', 'Best ANN'], fontsize=fsize)
    plt.xlim(min([config['acc'] for config in configs + configs_p]) - 1, 96.07)

    # Plot the axis descriptors
    plt.ylabel('J/inference', fontsize=fsize)
    plt.yscale('log')
    plt.yticks([max([float(config['snn_spinnaker_total_energy']) for config in configs + configs_p]),
                min([float(config['snn_spinnaker_total_energy']) for config in configs + configs_p]),
                max([float(config['snn_spinnaker2_total_energy']) for config in configs + configs_p]),
                min([float(config['snn_spinnaker2_total_energy']) for config in configs + configs_p]),
                max([float(config['snn_loihi_total_energy']) for config in configs + configs_p]),
                min([float(config['snn_loihi_total_energy']) for config in configs + configs_p]),
                float(configs[0]['ann_cpu_total_energy']),
                3 * float(configs[0]['ann_gpu_total_energy']),
                float(configs[0]['ann_gpu_total_energy']),
                float(configs[0]['ann_loihi_total_energy'])
                ],
               ['{:.6f}'.format(max([float(config['snn_spinnaker_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(min([float(config['snn_spinnaker_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(max([float(config['snn_spinnaker2_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(min([float(config['snn_spinnaker2_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(max([float(config['snn_loihi_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(min([float(config['snn_loihi_total_energy']) for config in configs + configs_p])),
                '{:.6f}'.format(float(configs[0]['ann_cpu_total_energy'])),
                '{:.6f}'.format(3 * float(configs[0]['ann_gpu_total_energy'])),
                '{:.6f}'.format(float(configs[0]['ann_gpu_total_energy'])),
                '{:.6f}'.format(float(configs[0]['ann_loihi_total_energy']))
                ],
               fontsize=fsize)
    secax = ax.secondary_yaxis('right', functions=(lambda x: x, lambda x: x))
    secax.set_ticks([float(configs[0]['ann_cpu_total_energy']),
                     3 * float(configs[0]['ann_gpu_total_energy']),
                     float(configs[0]['ann_gpu_total_energy']),
                     float(configs[0]['ann_loihi_total_energy'])
                     ])
    secax.set_yticklabels(
        ['ANN CPU\ntotal energy',
         'ANN ARM\ntotal energy',
         'ANN GPU\ntotal energy',
         'ANN Loihi\ntotal energy\n(hypothetical)'
         ], fontsize=fsize)

    plt.legend(loc='lower right', fontsize=fsize)

    plt.show()


# - Spikes visualization - #
def plot_spikes(path_to_save: str,
                examples: tuple,
                start: int,
                stop: int,
                labels: list,
                simulator,
                data,
                probe,
                network_output,
                n_steps: int,
                scale: float = 1.,
                show=False):
    """
    Plots the spike activity, given the input.

    Parameters
    ----------
    path_to_save : str
        Path to where to save the file.
    examples :
        First element of the tuple are the input images, the second is the output label index
    start : int
        Starting index for the examples to display. Must be non-negative
    stop : int
        Stopping index for the examples to display. Must have stop > start.
    labels : list
        List of the output labels.
    simulator :
        NengoDL simulator (https://www.nengo.ai/nengo-dl/reference.html#nengo_dl.Simulator)
    data :
        Simulator predictions, given the input.
    probe :
        Nengo probe (https://www.nengo.ai/nengo/frontend-api.html#nengo.Probe).
    network_output :
        Outputs of Keras output layer converted by Nengo converter.
    n_steps : int
        Number of time steps of the simulation
    scale : float
        Scale factor for the rate of spikes.
    show : bool
        Whether to show the plots.
    """

    assert stop > start >= 0

    # Unpack the data
    x, y = examples
    num_examples = stop - start

    # plot the results
    fig = plt.figure(figsize=(30, 10 * num_examples), tight_layout=True)
    for i in range(num_examples):
        # Input image
        plt.subplot(num_examples, 3, 3 * i + 1)
        plt.imshow(x[start + i])
        plt.title(labels[y[start + i]], fontsize=24)
        plt.axis('off')

        # Sample layer activations
        plt.subplot(num_examples, 3, 3 * i + 2)
        scaled_data = data[probe][start + i] * scale
        scaled_data *= 0.001
        plt.plot(range(1, len(scaled_data) + 1), scaled_data)
        rates = np.sum(scaled_data, axis=0) / (n_steps * simulator.dt)
        plt.ylabel('Number of spikes', fontsize=24)
        plt.title(f'Sample layer neural activities (mean={rates.mean():.1f} Hz, max={rates.max():.1f} Hz)',
                  fontsize=24)

        # Output predictions
        plt.subplot(num_examples, 3, 3 * i + 3)
        plt.plot(range(1, len(scaled_data) + 1), tf.nn.softmax(data[network_output][start + i]))
        plt.title('Output predictions', fontsize=24)
        plt.legend(labels, loc='upper left', fontsize=16)
        plt.xlabel('Timestep', fontsize=24)
        plt.ylabel('Probability', fontsize=24)

    # Make the directory for the figures
    try:
        os.mkdir('figs')
    except FileExistsError:
        pass

    # Save the figure
    plt.savefig(path_to_save)

    # Show the figure
    if show:
        plt.show()

    # And close it
    plt.close(fig=fig)


# - Accuracy visualization - #
def plot_timestep_accuracy(synapses: list,
                           scales: list,
                           timesteps: list,
                           accuracies,
                           x_logscale: bool = False,
                           y_logscale: bool = False,
                           show: bool = False):
    """
    Plots the accuracy against the number of time steps, with respect to different levels of synapse and firing rate.

    Parameters
    ----------
    synapses : list
        List of synapse values. Must be non-empty.
    scales : list
        List of firing rate scaling factors. Must be non-empty.
    timesteps :
        List of time steps. Must be non-empty.
    accuracies : ndarray
        Accuracy level corresponding to respective parameters.
        Axis 0 corresponds to synapses, axis 1 to firing rate scales, axis 2 to timesteps.
    x_logscale : bool
        `True` if the x-axis should be logarithmic.
    y_logscale : bool
        `True` if the y-axis should be logarithmic.
    show : bool
        Whether to display the plot.
    """

    # Markers and colours
    markers = ['v', 'o', '*', 's', 'P']
    colours = ['m', 'c', 'r', 'g', 'b']

    # Assertions
    assert 0 < len(synapses) == accuracies.shape[0] <= len(markers)
    assert 0 < len(scales) == accuracies.shape[1] <= len(colours)
    assert 0 < len(timesteps) == accuracies.shape[2]
    assert 0. <= np.min(accuracies) < np.max(accuracies) <= 1.

    # Figure
    fig = plt.figure(figsize=(3 * len(timesteps), 10), tight_layout=True)

    # Plot data
    for n in range(len(scales)):
        colour = colours[n]
        for m in range(len(synapses)):
            marker = markers[m]
            plt.plot(timesteps, accuracies[m, n], colour + marker + ':', markersize=12,
                     label='Scale: {}, synapse: {}'.format(scales[n], synapses[m]))

    # Format the plot
    # - Axes labels
    plt.xlabel('Time steps')
    plt.ylabel('Accuracy')

    # - Ticks
    plt.xticks(timesteps, timesteps)
    y_ticks = np.linspace(np.min(accuracies), np.max(accuracies), 10)

    # - Log scale conditionals
    if x_logscale:
        plt.xscale('log')
        plt.xlabel('Time steps (log scale)')
    if y_logscale:
        assert 0. < np.min(accuracies)
        plt.yscale('log')
        plt.ylabel('Accuracy (log scale)')
        y_ticks = np.logspace(np.min(accuracies), np.max(accuracies), 10, base=np.min(accuracies))

    # - Ticks again
    plt.yticks(y_ticks, ['{:.0f}%'.format(100 * n) for n in y_ticks])

    # - Miscellaneous
    plt.title('Time-step accuracy for selected synapses and firing rate factors')
    plt.legend()
    plt.grid()

    # Make the directory for the plot
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass

    # Save the figure
    plt.savefig('./plots/timestep_acc.png')

    # Display the figure
    if show:
        plt.show()

    # And close it
    plt.close(fig=fig)


# - Data visualization - #
def visualize_data(data, class_names: list):
    """
    Visualizes input data images.
    Parameters
    ----------
    data :
        tf.dataset object with images and labels batched.
    class_names : list
        List of strings corresponding to class names of the dataset.
    """

    # Make figure
    plt.figure(figsize=(10, 10))

    for images, labels in data.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i].numpy()])
            plt.axis('off')

    # Display
    plt.show()
