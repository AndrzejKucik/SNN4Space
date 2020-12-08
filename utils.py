#!/usr/bin/env python3.6

"""Utility functions."""

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.1'
__date__ = '2020-12-08'


# - Image processing and augmentation - #
def rescale_resize(image, image_size: tuple):
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


def augment(image,
            image_size: tuple,
            max_brightness_delta: float = 0.,
            max_hue_delta: float = 0.,
            lower_contrast: float = .999,
            upper_contrast: float = 1.,
            lower_saturation: float = .999,
            upper_saturation: float = 1.):
    """
    Image augmantation function.

    Parameters
    ----------
    image :
        3-D Tensor of shape [height, width, 3] and with non-negative integer values.
    image_size : tuple
        Tuple of 2 elements: new_height, new_width. The new size for the images.
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

    """
    # Rescale
    image = tf.cast(image, tf.float32) / 255.

    # Resize and random crop
    image = tf.image.resize_with_crop_or_pad(image, image_size[0] + 6, image_size[1] + 6)
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
    Plotes the spike activity, given the input.

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

    # Unpack data
    x, y = examples
    num_examples = stop - start

    # plot the results
    fig = plt.figure(figsize=(30, 10 * num_examples), tight_layout=True)
    for i in range(start, stop):
        # Input image
        plt.subplot(num_examples, 3, 3 * i + 1)
        plt.imshow(x[i])
        plt.title(labels[y[i]], fontsize=24)
        plt.axis('off')

        # Sample layer activations
        plt.subplot(num_examples, 3, 3 * i + 2)
        scaled_data = data[probe][i] * scale
        scaled_data *= 0.001
        plt.plot(range(1, len(scaled_data) + 1), scaled_data)
        rates = np.sum(scaled_data, axis=0) / (n_steps * simulator.dt)
        plt.ylabel('Number of spikes', fontsize=24)
        plt.title(f'Sample layer neural activities (mean={rates.mean():.1f} Hz, max={rates.max():.1f} Hz)',
                  fontsize=24)

        # Output predictions
        plt.subplot(num_examples, 3, 3 * i + 3)
        plt.plot(range(1, len(scaled_data) + 1), tf.nn.softmax(data[network_output][i]))
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

    # Show
    if show:
        plt.show()

    plt.close(fig=fig)
