#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Loads land cover datasets: UC Merced or EuroSAT."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.1'
__date__ = '2022-01-28'

# -- Third-party modules -- #
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

# -- Proprietary modules -- #
from utils import colour_str

# Default augmentation parameters
AUGMENTATION_PARAMETERS = {'lower_zoom': .95,
                           'upper_zoom': 1.05,
                           'max_brightness_delta': .2,
                           'max_hue_delta': .1,
                           'lower_contrast': .2,
                           'upper_contrast': 1.8,
                           'lower_saturation': .9,
                           'upper_saturation': 1.1}


def add_temporal_dim(timesteps: int = 1):
    """Repeats the image along the temporal dimension (Applied before batching)."""

    return lambda image, label: (tf.repeat(tf.expand_dims(image, axis=0), timesteps, axis=0), label)


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
        New image size: (new_height, new_width)
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

    for parameter in AUGMENTATION_PARAMETERS.keys():
        assert parameter in augmentation_parameters.keys(), colour_str('Augmentation parameter not understood!', 'red')

    def _augment(image, label):
        image = augment_image(image=image, image_size=image_size, **augmentation_parameters)
        return image, label

    return _augment


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


def load_data(dataset: str = 'eurosat',
              input_size: tuple = (64, 64),
              augmentation_parameters=None,
              batch_size: int = 32,
              timesteps: int = 0):
    """
    Dataloader.

    Parameters
    ----------
    dataset : str
        Name of the dataset. Either 'eurosat' or 'ucm'. Can also contain 'prewitt', 'sobel', and 'sq' if a filter
        input map is to be applied.
    input_size : tuple
        Size of input images: (height, width)
    augmentation_parameters : dict
        Augmentation parameters values.
    batch_size : int
        Batch size.
    timesteps : int
        Simulation timesteps for SNN (optional).
    Returns
    -------
    train :
        Training dataset.
    val :
        Validation dataset.
    test :
        Test dataset
    info :
        Dataset info.
    """
    if augmentation_parameters is None:
        augmentation_parameters = AUGMENTATION_PARAMETERS

    # Load data
    if 'ucm' in dataset:
        print(f"Training on {colour_str('UC Merced', 'blue')} dataset",
              "(http://weegee.vision.ucmerced.edu/datasets/landuse.html)")
        (train, val, test), info = tfds.load('uc_merced',
                                             split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                             with_info=True,
                                             as_supervised=True)
    else:  # eurosat
        print(f"Training on {colour_str('EurosatRGB', 'blue')} dataset (https://github.com/phelber/EuroSAT)")
        (train, val, test), info = tfds.load('eurosat/rgb',
                                             split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                             with_info=True,
                                             as_supervised=True)

    # Prepare for training
    # - Resize and rescale the images, and cache the training and the validation sets for faster training
    train = train.map(rescale_resize(image_size=input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    val = val.map(rescale_resize(image_size=input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    test = test.map(rescale_resize(image_size=input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Shuffle the training set and apply the augmentation (after caching to avoid caching randomness)
    num_train = int(info.splits['train'].num_examples * .8)
    train = train.shuffle(num_train).map(augment(image_size=input_size,
                                                 augmentation_parameters=augmentation_parameters),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Add temporal dimension (only for SNN)
    if timesteps > 0:
        train = train.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val = val.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.map(add_temporal_dim(timesteps=timesteps), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # -- Optional gradient-based input (Prewitt and Sobel filters must be to 4D tensors)
        train = train.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val = val.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Batch data
    train = train.batch(batch_size)
    val = val.batch(batch_size)
    test = test.batch(batch_size)

    if timesteps == 0:
        # - Optional gradient-based input (Prewitt and Sobel filters must be to 4D tensors
        train = train.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val = val.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.map(input_filter_map(filter_name=dataset), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # - Prefetch data
    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)

    return train, val, test, info


def rescale_resize_image(image, image_size: tuple):
    """
    Converts an integer image tensor to a float,scales it down it to [0, 1], and resizes to a desired size.

    Parameters
    ----------
    image :
        3-D Tensor of shape [height, width, channels] and with non-negative integer values.
    image_size : tuple
        Size for the new image: (new_height, new_width).

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
