#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Argument parser."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.1'
__date__ = '2022-01-28'

# -- Built-in modules -- #
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs

# - Third-party modules -- #
import tensorflow as tf


def parse_arguments(arguments: list):
    """
    Parses arguments.
    
    Parameters
    ----------
    arguments : List
        Arguments for parsing.

    Returns
    -------
    arguments : dict
        Dictionary of parsed arguments.

    """

    # Argument parser
    parser = ArgumentParser()

    # - Parameters
    # -- Random seed (fro reproducibility)
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=5,
                        help='Global random seed.')
    # -- Dataset
    parser.add_argument('-ds',
                        '--dataset',
                        type=str,
                        default='eurosat',
                        help='Dataset. Either `eurosat` or `ucm`. One can also add `prewitt`, `sobel`, `mask` or `sq`.')
    # -- Models
    parser.add_argument('-md',
                        '--model_path',
                        type=str,
                        help='Path to pretrained ANN model. Make sure that it is consistent with the dataset.')
    parser.add_argument('-wp',
                        '--weights_path',
                        type=str,
                        help='Path to weights of a pretrained SNN.')
    # -- Training
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=1000,  # 16 SNN
                        help='(Target) number of epochs.')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=32,  # 16 SNN
                        help='(Target batch) size (per replica).')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=.001,  # 3e-5 SNN
                        help='(Target) learning rate.')
    # --- ANN
    parser.add_argument('-kl2',
                        '--kernel_l2',
                        type=float,
                        default=1e-4,
                        help='Regularization L2 parameter for the convolutional kernels.')
    parser.add_argument('-bl1',
                        '--bias_l1',
                        type=float,
                        default=1e-5,
                        help='Regularization L1 parameter for the convolutional biases.')
    # --- SNN
    # --- To speed-up the training, we may do it iteratively, by increasing the number of timesteps by a factor of 2,
    # --- starting with a single timestep. We can also start with large batch sizes, and decrease them (also by a factor
    # --- of 2), but then we must remember to lower the learning rate. The other parameters are treated as targets, so
    # --- they will be used in the final training step.
    parser.add_argument('-i',
                        '--iterate',
                        action='store_true',
                        default=False,
                        help='If `True`, then the training is iterative.')
    parser.add_argument('-t',
                        '--timesteps',
                        type=int,
                        default=32,
                        help='Target number of simulation timesteps (the training starts with 1).')
    parser.add_argument('-dt',
                        '--dt',
                        type=float,
                        default=1.,
                        help='Simulation temporal resolution. Decayed during SNN training.')
    parser.add_argument('-l2',
                        '--l2',
                        type=float,
                        default=1e-9,
                        help='L2 regularization for the spike frequencies.')
    parser.add_argument('-lhz',
                        '--lower_hz',
                        type=float,
                        default=10.,
                        help='Lower frequency target for the spikes (Hz). Must be positive')
    parser.add_argument('-uhz',
                        '--upper_hz',
                        type=float,
                        default=20.,
                        help='Upper frequency target for the spikes (Hz). Must be bigger than lower_hz')
    parser.add_argument('-tau',
                        '--tau',
                        type=float,
                        default=0.1,
                        help='Tau parameter for the low-pass filter.')
    # -- Augmentation
    parser.add_argument('-lz',
                        '--lower_zoom',
                        type=float,
                        default=.95,
                        help='Augmentation parameter. Lower bound for a random zoom factor. Must be positive.')
    parser.add_argument('-uz',
                        '--upper_zoom',
                        type=float,
                        default=1.05,
                        help='Augmentation parameter. Upper bound for a random zoom factor. '
                             + 'Must be bigger than lower_zoom.')
    parser.add_argument('-mbd',
                        '--max_brightness_delta',
                        type=float,
                        default=.2,
                        help='Augmentation parameter. Maximum brightness delta. Must be a non-negative float.')
    parser.add_argument('-mhd',
                        '--max_hue_delta',
                        type=float,
                        default=.1,
                        help='Augmentation parameter. Maximum hue delta. Must be in the interval [0, .5].')
    parser.add_argument('-lc',
                        '--lower_contrast',
                        type=float, default=.2,
                        help='Augmentation parameter. Lower bound for a random contrast factor. Must be positive.')
    parser.add_argument('-uc',
                        '--upper_contrast',
                        type=float,
                        default=1.8,
                        help='Augmentation parameter. Upper bound for a random contrast factor. '
                             + 'Must be bigger than lower_contrast.')
    parser.add_argument('-ls',
                        '--lower_saturation',
                        type=float,
                        default=.9,
                        help='Augmentation parameter. Lower bound for a random saturation factor. Must be positive.')
    parser.add_argument('-us',
                        '--upper_saturation',
                        type=float,
                        default=1.1,
                        help='Augmentation parameter. Upper bound for a random saturation factor. '
                             + 'Must be bigger than lower_saturation.')
    # -- Energy estimation verbosity
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        default=False,
                        help='If True, the energy contributions for all the layers is shown.')

    # - Parse arguments
    arguments = vars(parser.parse_args(arguments))

    # Assertions
    assert 0 < arguments['lower_hz'] < arguments['upper_hz']
    assert 0 < arguments['lower_zoom'] < arguments['upper_zoom']
    assert 0 <= arguments['max_brightness_delta']
    assert 0 <= arguments['max_hue_delta'] <= .5
    assert 0 < arguments['lower_contrast'] < arguments['upper_contrast']
    assert 0 < arguments['lower_saturation'] < arguments['upper_saturation']

    # Log time
    arguments['time'] = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Dataset
    arguments['dataset'] = arguments['dataset'].lower()
    arguments['input_shape'] = (64, 64, 3) if 'eurosat' in arguments['dataset'] else (224, 224, 3)  # ucm
    arguments['num_classes'] = 10 if 'eurosat' in arguments['dataset'] else 21  # ucm

    # Model path
    if arguments['model_path'] is None:  # For ANN training
        arguments['model_name'] = f"{arguments['dataset']}/{arguments['time']}"
        arguments['model_path'] = f"models/{arguments['model_name']}"
        makedirs(arguments['model_path'], exist_ok=True)
    else:  # For SNN training
        if arguments['weights_path'] is None:  # - First time training
            arguments['model_name'] = f"{arguments['dataset']}_spiking/{arguments['time']}"
            arguments['weights_path'] = f"models/{arguments['model_name']}"
            makedirs(arguments['weights_path'], exist_ok=True)
        else:  # - Fine-tuning
            arguments['model_name'] = f"{arguments['dataset']}_spiking_fine_tuned/{arguments['time']}"

    # Log the arguments to Tensorboard
    summary_writer = tf.summary.create_file_writer(f"logs/{arguments['model_name']}/arguments")
    with summary_writer.as_default():
        for key, value in arguments.items():
            if isinstance(value, str):
                tf.summary.text(key, value, step=0)
            elif isinstance(value, (int, float)):
                tf.summary.scalar(key, value, step=0)
            elif isinstance(value, (tuple, list)):
                for n in range(len(value)):
                    tf.summary.scalar(key, value[n], step=n)

    return arguments
