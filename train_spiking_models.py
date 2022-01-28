#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Spiking aware fine-tuning of the VGG16 network trained on the EuroSAT RGB or UC Merced datasets."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.3.2'
__date__ = '2022-01-28'

# -- Built-in modules -- #
import sys

# -- Third-party modules -- #
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from argument_parser import parse_arguments
from create_models import create_spiking_vgg16_model
from dataloaders import AUGMENTATION_PARAMETERS, load_data
from utils import colour_str, DTStop

# Target dt
DT_TARGET = .001  # 1ms

if __name__ == '__main__':
    # Get the arguments
    args = parse_arguments(arguments=sys.argv[1:])

    # Set the seed for reproducibility
    tf.random.set_seed(seed=args['seed'])

    # Strategy parameters (for multiple GPU training) #
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    num_devices = strategy.num_replicas_in_sync
    print(f"Number of devices: {colour_str(num_devices, 'purple')}")

    # The model
    # - Model path
    with strategy.scope():
        # - A variable to be passed to the model as the simulation time resolution
        dt_var = tf.Variable(args['dt'], aggregation=tf.VariableAggregation.MEAN)

        # - Create a model
        model = create_spiking_vgg16_model(model_path=args['model_path'],
                                           input_shape=args['input_shape'],
                                           dt=dt_var,
                                           l2=args['l2'],
                                           lower_hz=args['lower_hz'],
                                           upper_hz=args['upper_hz'],
                                           tau=args['tau'],
                                           num_classes=args['num_classes'],
                                           spiking_aware_training=True)


        # - Functions to monitor the variables
        # noinspection PyUnusedLocal
        def dt_monitor(y_true, y_pred):
            return dt_var.read_value()


        # - Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(args['learning_rate']),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy(), dt_monitor])

    # Show model's summary
    model.summary()

    # Iterate or not
    # Get the exponents for scaling by a factor of 2.
    exponent = int(np.log2(args['timesteps']))
    start = 0 if args['iterate'] else exponent
    for n in range(start, exponent + 1):
        # Iterative hyperparameters
        timesteps = 2 ** n
        batch_size = 2 ** (exponent - n) * args['batch_size'] * num_devices
        lr = args['learning_rate'] * 2 ** (exponent - n)
        tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
        epochs = args['epochs'] * 2 ** (exponent - n)

        # Load data
        augmentation_parameters = {key: args[key] for key in AUGMENTATION_PARAMETERS.keys()}
        train, val, test, _ = load_data(dataset=args['dataset'],
                                        input_size=args['input_shape'][:-1],
                                        augmentation_parameters=augmentation_parameters,
                                        batch_size=batch_size,
                                        timesteps=timesteps)

        # Callbacks
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f"logs/{args['model_name']}/{n}/fit", histogram_freq=1),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=epochs // 4, verbose=True),
                     tf.keras.callbacks.EarlyStopping(patience=epochs // 2, verbose=True)]

        if dt_var.value() > DT_TARGET:
            callbacks.append(DTStop(dt=dt_var, dt_min=DT_TARGET))

        # Print the training iteration parameters
        print(f"Starting the training for {colour_str(epochs, 'orange')} epoch(s),"
              f"with {colour_str(timesteps, 'orange')} timestep(s)",
              f"on batches of {colour_str(batch_size, 'orange')} example(s),"
              f"and the learning rate {colour_str(lr, 'orange')}.")

        # Train the model
        print(f'Commencing the training on iteration', colour_str(f'{min(n + 1, exponent)}/{exponent}', 'orange') + '.')
        model.fit(x=train, epochs=epochs, validation_data=val, callbacks=callbacks)

        # Evaluate the model
        results = model.evaluate(x=test, batch_size=batch_size, verbose=True)
        try:
            loss, acc, dt_stop = results
        except ValueError:
            loss, acc = results
            dt_stop = DT_TARGET

        print("Model's accuracy:", colour_str(f'{acc:.2%}', 'green'))

        # Log the evaluation results to Tensorboard
        summary_writer = tf.summary.create_file_writer(f"logs/{args['model_name']}/{n}/evaluate")
        with summary_writer.as_default():
            tf.summary.scalar('Test loss', loss, step=args['epochs'])
            tf.summary.scalar('Test accuracy', acc, step=args['epochs'])
            tf.summary.scalar('Final dt', dt_stop, step=args['epochs'])

        # New model to avoid serialization issues
        with strategy.scope():
            new_model = create_spiking_vgg16_model(model_path=args['model_path'],
                                                   input_shape=args['input_shape'],
                                                   dt=dt_stop,
                                                   l2=args['l2'],
                                                   lower_hz=args['lower_hz'],
                                                   upper_hz=args['upper_hz'],
                                                   tau=args['tau'],
                                                   num_classes=args['num_classes'],
                                                   spiking_aware_training=True)

            # - Load weights (skipping dt)
            new_model.set_weights([w for w in model.get_weights() if w.shape != ()])

            # - Compile the model
            new_model.compile(optimizer=tf.keras.optimizers.RMSprop(args['learning_rate']),
                              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=[tf.metrics.SparseCategoricalAccuracy()])

        # Save model filepath
        new_model.save(f"{args['weights_path']}/{n}.h5")

        # We stop optimising dt here
        if dt_stop <= DT_TARGET:
            model = new_model

        del new_model
