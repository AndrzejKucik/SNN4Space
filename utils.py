#!/usr/bin/env python3.6

"""Utility functions."""

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__contributors__ = 'Gabriele Meoni'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.11'
__date__ = '2021-02-25'

# Colour dictionary
COLOUR_DICTIONARY = {'red': '\033[0;31m',
                     'black': '\033[0m',
                     'green': '\033[0;32m',
                     'orange': '\033[0;33m',
                     'purple': '\033[0;35m',
                     'blue': '\033[0;34m',
                     'cyan': '\033[0;36m'}


class DTStop(tf.keras.callbacks.Callback):
    """Stops dt updates after it has reach the desired minimum value (dt_min)."""

    def __init__(self, dt, dt_min: float = 0.001):
        super(DTStop).__init__()
        self.dt = dt
        self.dt_min = dt_min

    def on_epoch_begin(self, epoch, logs=None):
        if self.dt.value() <= self.dt_min:
            self.dt.assign(self.dt_min)


# - Image preprocessing and augmentation - #
def rescale_resize_image(image, image_size: tuple):
    """
    Converts an integer image tensor to a float,scales it down it to [0, 1], and resizes to a desired size.

    Parameters
    ----------
    image :
        3-D Tensor of shape [height, width, channels] and with non-negative integer values.
    image_size : tuple
        Tuple of 2 elements: new_height, new_width. The new size for the images.

    Returns
    -------
    image :
        3-D Tensor of shape [new_height, new_width, channels].
    """

    # Rescale
    image = tf.cast(image, tf.float32) / 255.

    # Resize
    image = tf.image.resize(image, image_size)

    return image


def rescale_resize(image_size):
    """
    Returns a function resizing an image to the desired size, and passing on the label.
    Parameters
    ----------
    image_size : tuple
        Image height and width.
    """

    return lambda image, label: (rescale_resize_image(image, image_size), label)


def augment_image(image,
                  image_size: tuple,
                  lower_zoom: float = .999,
                  upper_zoom: float = 1.,
                  max_brightness_delta: float = 0.,
                  max_hue_delta: float = 0.,
                  lower_contrast: float = .999,
                  upper_contrast: float = 1.,
                  lower_saturation: float = .999,
                  upper_saturation: float = 1.):
    """
    Image augmentation function.

    Parameters
    ----------
    image :
        3-D Tensor of shape [height, width, 3] and with non-negative integer values.
    image_size : tuple
        Tuple of 2 elements: new_height, new_width. The new size for the images.
    lower_zoom : float
        Lower bound for a random zoom factor. Must be positive.
    upper_zoom : float
        Upper bound for a random zoom factor. Must be bigger than lower_zoom.
        Note: Zoom is applied to width and height independently.
    max_brightness_delta : float
        To adjust brightness by a delta randomly picked in the interval [-max_delta, max_delta). Must be non-negative.
    max_hue_delta : float
        To adjust hue by a delta randomly picked in the interval [-max_delta, max_delta).
        Must be in the interval [0., .5].
    lower_contrast : float
        Lower bound for a random contrast factor. Must be positive.
    upper_contrast : float
        Upper bound for a random contrast factor. Must be bigger than lower_contrast.
    lower_saturation : float
        Lower bound for a random saturation factor. Must be positive.
    upper_saturation : float
        Upper bound for a random saturation factor. Must be bigger than lower_saturation.

    Returns
    -------
    image :
        3-D Tensor of shape [height, width, 3] and with non-negative integer values.
    """

    # Random zoom
    zoom = tf.random.uniform((2,), minval=lower_zoom, maxval=upper_zoom)
    image = tf.image.resize(image, [int(zoom[0] * image_size[0]), int(zoom[1] * image_size[1])])

    # Random crop
    image = tf.image.resize_with_crop_or_pad(image, int(1.03 * image_size[0]), int(1.03 * image_size[1]))
    image = tf.image.random_crop(image, size=[image_size[0], image_size[1], 3])

    # Random flip
    image = tf.image.random_flip_left_right(image)

    # Random rotation
    image = tf.image.rot90(image, k=tf.cast(tf.random.uniform(shape=(1,)) * 4, tf.int32)[0])

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=max_brightness_delta)

    # Random contrast
    image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)

    # Random hue
    image = tf.image.random_hue(image, max_delta=max_hue_delta)

    # Random saturation
    image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)

    # Clip
    image = tf.clip_by_value(image, 0, 1)

    return image


def augment(image_size: tuple, augmentation_parameters: dict):
    """
    Returns a function applying augmentation to input images and passing on their labels.

    Parameters
    ----------
    image_size : tuple
        Height and width of an input image.
    augmentation_parameters : dict
        Dictionary with values to be passed to the augmentation function as arguments.

    Returns
    -------
    _augment : lambda
        Augmentation function.
    """

    for parameter in ['lower_zoom', 'upper_zoom', 'max_brightness_delta', 'max_hue_delta',
                      'lower_contrast', 'upper_contrast', 'lower_saturation', 'upper_saturation']:
        assert parameter in augmentation_parameters.keys()

    def _augment(image, label):
        image = augment_image(image=image,
                              image_size=image_size,
                              lower_zoom=augmentation_parameters['lower_zoom'],
                              upper_zoom=augmentation_parameters['upper_zoom'],
                              max_brightness_delta=augmentation_parameters['max_brightness_delta'],
                              max_hue_delta=augmentation_parameters['max_hue_delta'],
                              lower_contrast=augmentation_parameters['lower_contrast'],
                              upper_contrast=augmentation_parameters['upper_contrast'],
                              lower_saturation=augmentation_parameters['lower_saturation'],
                              upper_saturation=augmentation_parameters['upper_saturation'])
        return image, label

    return _augment


# - Input filters - #
def input_filter_map(filter_name: str):
    """
    Function returning a function applying a filter to the input images and passing on the label.
    Parameters
    ----------
    filter_name : str
        Name of an input filter, works with `prewitt`, `sobel`, `mask`, and `sq`.
    Returns
    -------
    image_filter : lambda
        Function taking a tensor tuple (images, label) as the input. Images are assumed to be batched.
    """

    def image_filter(images, label):
        if 'prewitt' in filter_name.lower():
            # Apply Prewitt filter and normalize
            new_images = tfio.experimental.filter.prewitt(images) / tf.sqrt(10.)
        elif 'sobel' in filter_name.lower():
            # Apply Sobel filter and normalize
            new_images = tfio.experimental.filter.sobel(images) / tf.sqrt(20.)
        else:
            new_images = images

        if 'sq' in filter_name.lower():
            # Square the input:
            new_images = new_images ** 2

        # Ignore small values
        new_images = new_images * tf.cast(new_images >= 1 / 255., tf.float32)

        # Apply filter mas to the original images
        if 'mask' in filter_name.lower():
            new_images = images * tf.cast(new_images > 0., tf.float32)

        return new_images, label

    return image_filter


def add_temporal_dim(timesteps: int = 1):
    """Repeats the image along the temporal dimension (Applied before batching)."""

    return lambda image, label: (tf.repeat(tf.expand_dims(image, axis=0), timesteps, axis=0), label)


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
