#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Utility functions."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__contributors__ = 'Gabriele Meoni'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.3.0'
__date__ = '2022-01-28'

# -- Built-in modules -- #
from pathlib import Path

# -- Third-party modules -- #
import tensorflow as tf

# Colour dictionary
COLOURS = {'red': '\033[0;31m',
           'black': '\033[0m',
           'green': '\033[0;32m',
           'orange': '\033[0;33m',
           'purple': '\033[0;35m',
           'blue': '\033[0;34m',
           'cyan': '\033[0;36m'}


def colour_str(word, colour: str):
    """Function to colour strings."""
    return COLOURS[colour.lower()] + str(word) + COLOURS['black']


class DTStop(tf.keras.callbacks.Callback):
    """Stops dt updates after it has reach the desired minimum value (dt_min)."""

    def __init__(self, dt, dt_min: float = .001):
        super(DTStop).__init__()
        self.dt = dt
        self.dt_min = dt_min

    def on_epoch_begin(self, epoch, logs=None):
        if self.dt.value() <= self.dt_min:
            self.dt.assign(self.dt_min)


# - Image preprocessing and augmentation - #


# - Input filters - #


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
