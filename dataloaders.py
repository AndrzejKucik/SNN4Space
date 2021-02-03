#!/usr/bin/env python3.6

"""Loads land cover datasets: UC Merced or EuroSAT."""

# -- Built-in modules -- #
import pathlib

# -- Third-party modules -- #
import tensorflow as tf
import tensorflow_datasets as tfds

# -- File info -- #
__author__ = 'Andrzej S. Kucik'
__copyright__ = 'European Space Agency'
__contact__ = 'andrzej.kucik@esa.int'
__version__ = '0.1.2'
__date__ = '2021-02-03'


def load_ucm():
    """Loads the UC Merced dataset (http://weegee.vision.ucmerced.edu/datasets/landuse.html)."""

    # Load data
    (ucm_train, ucm_val, ucm_test), info = tfds.load('uc_merced',
                                                     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                     with_info=True,
                                                     as_supervised=True)

    # Number of examples
    num_train = int(info.splits['train'].num_examples * .8)
    num_val = int(info.splits['train'].num_examples * .1)
    num_test = int(info.splits['train'].num_examples * .1)
    print('Number of training examples: {}, validation examples: {}, and test examples: {}'.format(num_train,
                                                                                                   num_val,
                                                                                                   num_test))

    return ucm_train, ucm_val, ucm_test, info.features['label'].names


def load_eurosat():
    """Loads the EuroSAT RGB dataset (https://github.com/phelber/EuroSAT)."""

    # Load data
    eurosat_path = tf.keras.utils.get_file(fname='eurosat_rgb',
                                           origin='http://madm.dfki.de/files/sentinel/EuroSAT.zip',
                                           extract=True)
    eurosat_path = pathlib.Path(eurosat_path).parent.joinpath('2750')

    # Number of examples
    num_examples = len(list(eurosat_path.glob('*/*.jpg')))
    num_train = int(num_examples * .8)
    num_val = int(num_examples * .1)
    num_test = int(num_examples * .1)
    print('Number of training examples: {}, validation examples: {}, and test examples: {}'.format(num_train,
                                                                                                   num_val,
                                                                                                   num_test))
    # Create a dataset from the files; shuffle seed set for reproducibility
    eurosat = tf.keras.preprocessing.image_dataset_from_directory(directory=eurosat_path,
                                                                  image_size=(64, 64),
                                                                  shuffle=True,
                                                                  seed=123)
    # Get the labels
    labels = eurosat.class_names

    # Unbatch before splitting into training, validation, and test sets
    eurosat = eurosat.unbatch()

    # Split into training, validation, and test sets
    eurosat_train = eurosat.take(num_train)
    eurosat_val = eurosat.skip(num_train)
    eurosat_test = eurosat_val.skip(num_val)
    eurosat_val = eurosat_val.take(num_val)

    return eurosat_train, eurosat_val, eurosat_test, labels
