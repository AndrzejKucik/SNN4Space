#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""Fine-tuning of the VGG16 network on the UC Merced or EuroSAT datasets."""

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.4.0'
__date__ = '2022-01-28'

# -- Built-in modules -- #
import sys

# -- Third-party modules -- #
import tensorflow as tf

# -- Proprietary modules -- #
from argument_parser import parse_arguments
from create_models import create_vgg16_model
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
    train, val, test, _ = load_data(dataset=args['dataset'],
                                    input_size=args['input_shape'][:-1],
                                    augmentation_parameters=augmentation_parameters,
                                    batch_size=batch_size)

    # The model
    with strategy.scope():
        # - Create model
        model = create_vgg16_model(input_shape=args['input_shape'],
                                   kernel_l2=args['kernel_l2'],
                                   bias_l1=args['bias_l1'],
                                   num_classes=args['num_classes'],
                                   remove_pooling=False,
                                   use_dense_bias=False)

        # -- Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(args['learning_rate']),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    # -- Show the summary of the model
    model.summary()

    # Callbacks
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f"logs/{args['model_name']}/fit", histogram_freq=1),
                 tf.keras.callbacks.ReduceLROnPlateau(verbose=True, patience=50),
                 tf.keras.callbacks.EarlyStopping(patience=100),
                 tf.keras.callbacks.ModelCheckpoint(filepath=args['model_path'], verbose=True, save_best_only=True)]

    # Train the model
    model.fit(x=train, epochs=args['epochs'], validation_data=val, callbacks=callbacks)

    # Load the best weights
    model.load_weights(args['model_path'])

    # Evaluate the model
    loss, acc = model.evaluate(x=test)
    print("Best model's accuracy:", colour_str(f'{acc:.2%}', 'green'))

    # Log the evaluation results to Tensorboard
    summary_writer = tf.summary.create_file_writer(f"logs/{args['model_name']}/evaluate")
    with summary_writer.as_default():
        tf.summary.scalar('Test loss', loss, step=args['epochs'])
        tf.summary.scalar('Test accuracy', acc, step=args['epochs'])
