#!/usr/bin/env python3.6

"""Utility functions."""

# -- Built-in modules -- #
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf
import tensorflow_io as tfio

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__contributors__ = 'Gabriele Meoni'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.0'
__date__ = '2021-04-28'

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
        new_images = new_images * tf.cast(new_images >= 2 / 255., tf.float32)

        # Apply filter mas to the original images
        if 'mask' in filter_name.lower():
            new_images = images * tf.cast(new_images > 0., tf.float32)

        return new_images, label

    return image_filter


def add_temporal_dim(timesteps: int = 1):
    """Repeats the image along the temporal dimension (Applied before batching)."""

    return lambda image, label: (tf.repeat(tf.expand_dims(image, axis=0), timesteps, axis=0), label)


def model_config_dict(path):
    """
    Function returning a dictionary with the model configuration inferred from the file name.

    Parameters
    ----------
    path :
        Path to the model, either string or pathlib.Path object.

    Returns
    -------
    config : dict
        Dictionary with model configuration.

    """

    # Convert to Path
    if type(path) == str:
        path = Path(path)

    config = path.stem.split(sep='_')
    config = {config[n]: config[n + 1] for n in range(0, len(config) - 1, 2)}

    return config
