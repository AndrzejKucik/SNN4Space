#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Fine-tunes a spiking neural network on fixed temporal resolution."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.2.1'
__date__ = '2022-01-28'

# -- Built-in modules -- #
import sys

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from argument_parser import parse_arguments
from create_models import create_spiking_vgg16_model
from dataloaders import AUGMENTATION_PARAMETERS, load_data
from utils import colour_str

if __name__ == '__main__':
    # Get the arguments
    args = parse_arguments(arguments=sys.argv[1:])

    # Set the seed for reproducibility
    tf.random.set_seed(seed=args['seed'])

    # Strategy parameters (for multiple GPU training) #
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_devices = strategy.num_replicas_in_sync
    print(f"Number of devices: {colour_str(num_devices, 'purple')}")

    # Global batch size
    batch_size = args['batch_size'] * num_devices

    # Load data
    augmentation_parameters = {key: args[key] for key in AUGMENTATION_PARAMETERS.keys()}
    train, val, test, info = load_data(dataset=args['dataset'],
                                       input_size=args['input_shape'][:-1],
                                       augmentation_parameters=augmentation_parameters,
                                       batch_size=batch_size,
                                       timesteps=args['timesteps'])

    # Model
    # - Model path
    with strategy.scope():
        # - Create a model
        model = create_spiking_vgg16_model(model_path=args['model_path'],
                                           input_shape=args['input_shape'],
                                           dt=args['dt'],
                                           l2=args['l2'],
                                           lower_hz=args['lower_hz'],
                                           upper_hz=args['upper_hz'],
                                           tau=args['tau'],
                                           num_classes=args['num_classes'],
                                           spiking_aware_training=True)

        # - Load the weights
        model.load_weights(filepath=args['weights_path'])

        # - Compile the model
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            args['learning_rate'],
            decay_steps=int(info.splits['train'].num_examples * .8) // batch_size,
            decay_rate=.9,
            staircase=True)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # - Model summary
    model.summary()

    # Callbacks
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f"logs/{args['model_name']}/fit", histogram_freq=1)]

    # Train the model
    model.fit(x=train, epochs=args['epochs'], validation_data=val, callbacks=callbacks)

    # Evaluate the model
    loss, acc, = model.evaluate(x=test, batch_size=batch_size, verbose=True)
    print("Model's accuracy:", colour_str(f'{acc:.2%}', 'green'))

    # Log the evaluation results to Tensorboard
    summary_writer = tf.summary.create_file_writer(f"logs/{args['model_name']}/evaluate")
    with summary_writer.as_default():
        tf.summary.scalar('Test loss', loss, step=args['epochs'])
        tf.summary.scalar('Test accuracy', acc, step=args['epochs'])

    # Save weights
    model.save(f"{args['weights_path']}/fine_tuned.h5")
