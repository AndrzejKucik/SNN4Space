import tensorflow as tf
import numpy as np

from alexnet import AlexNet
import nengo
import nengo_dl

from tqdm import tqdm
import os
from PIL import Image
import numpy as np


def random_shuffle(img, labels):
    assert len(img) == len(labels)
    p = np.random.permutation(len(img))
    img = np.array(img)
    labels = np.array(labels)
    return img[p], labels[p]


def extract_ucm(ucm_path, img_shape, train_percentage, valid_percentage):
    classes = [
        name
        for name in os.listdir(ucm_path)
        if os.path.isdir(os.path.join(ucm_path, name))
    ]
    n_classes = len(classes)
    label_dict = dict(zip(classes, [n for n in range(len(classes))]))
    if n_classes <= 0:
        print("Error, not files found in", ucm_path)
        exit()
    data, label = [], []

    for ucm_class in tqdm(classes):
        print("Found class: ", ucm_class)
        ucm_class_path = os.path.join(ucm_path, ucm_class)
        files = [
            name
            for name in os.listdir(ucm_class_path)
            if os.path.isfile(os.path.join(ucm_class_path, name))
        ]

        for img_name in tqdm(files):
            img_name = os.path.join(ucm_class_path, img_name)
            img = Image.open(img_name)
            label.append(label_dict[ucm_class])
            img = img.resize(img_shape)
            data.append(np.array(img))
    label = np.array(label)
    data = np.array(data)
    data, label = random_shuffle(data, label)
    train_n_img = int(np.floor(train_percentage * len(label)))
    valid_n_img = int(np.floor(valid_percentage * len(label)))
    train_data = data[:train_n_img]
    train_label = label[:train_n_img]
    valid_data = data[train_n_img : train_n_img + valid_n_img]
    valid_label = label[train_n_img : train_n_img + valid_n_img]
    test_data = data[train_n_img + valid_n_img :]
    test_label = label[train_n_img + valid_n_img :]
    return train_data, train_label, valid_data, valid_label, test_data, test_label


def extract_dataset(
    dataset_path="./data/01_unprocessed",
    train_percentage=0.8,
    test_percentage=0.15,
    batch_size=32,
    img_size=(410, 389),
):
    print("Extracting training data...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=(1 - train_percentage),
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    print("Extracting validation data...")
    valid_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=(1 - train_percentage),
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    num_train_images = 0
    num_train_batches = 0
    for x, y in train_ds:
        num_train_images += len(x)
        num_train_batches += 1

    num_valid_test_batches = 0
    for x, y in valid_test_ds:
        num_valid_test_batches += 1

    valid_percentage = 1 - (train_percentage + test_percentage)
    total_batches = num_train_batches + num_valid_test_batches
    num_valid_batches = np.floor(valid_percentage * total_batches)
    valid_ds = valid_test_ds.take(num_valid_batches)
    test_ds = valid_test_ds.skip(num_valid_batches)
    return [train_ds, valid_ds, test_ds]


def create_model(
    STRATEGY,
    model_name="vgg16",
    input_shape=(224, 224, 3),
    num_classes=21,
    weights_path=None,
    kernel_l2=1e-4,
    bias_l1=1e-5,
    dropout=0,
    lr=0.01,
):
    """Creates a Keras model which is a modified version of the VGG16 network."""

    if model_name == "vgg16":
        # Load the VGG16 model (this may take a moment for the first time)
        original_model = tf.keras.applications.VGG16(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif model_name == "alexnet":
        original_model = AlexNet(input_shape=input_shape, num_classes=num_classes)

    with STRATEGY.scope():
        # Create new model
        # - Input layer
        input_layer = tf.keras.Input(shape=input_shape)

        # - First convolutional layer
        if model_name == "alexnet":
            config = original_model.layers[0].get_config()
        else:
            config = original_model.layers[1].get_config()

        filters, kernel_size, name, padding = (
            config["filters"],
            config["kernel_size"],
            config["name"],
            config["padding"],
        )
        # -- We keep the same number of filters and the kernel size but change the activation format and add
        # -- kernel and bias regularizers
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
            bias_regularizer=tf.keras.regularizers.l1(bias_l1),
            name=name,
        )(input_layer)

        # - The remaining layers
        for layer in original_model.layers[2:-1]:
            try:
                config = layer.get_config()
                filters, kernel_size, name, padding = (
                    config["filters"],
                    config["kernel_size"],
                    config["name"],
                    config["padding"],
                )
                x = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
                    bias_regularizer=tf.keras.regularizers.l1(bias_l1),
                    name=name,
                )(x)
            except KeyError:
                # -- Change the max pooling to average pooling
                x = tf.keras.layers.AveragePooling2D((2, 2))(x)
                # -- Add dropout if necessary
                if dropout > 0.0:
                    x = tf.keras.layers.Dropout(dropout)(x)

        # - Conclude VGG layers with a global average pooling layer
        global_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

        # - Output layer
        output_layer = tf.keras.layers.Dense(num_classes, use_bias=False)(global_pool)

        # - Define the model
        model = tf.keras.Model(input_layer, output_layer)

        # if weights_path is not None:
        #    # - After the model is defined, we can load the weights
        #    for layer in model.layers:
        #        if isinstance(layer, tf.keras.layers.Conv2D):
        #            weights = original_model.get_layer(name=layer.name).get_weights()
        #            layer.set_weights(weights)
        #
        # - Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(lr),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.SparseCategoricalAccuracy(),
        )

        if weights_path is not None:
            try:
                print("Loading weights...")
                model.load_weights(weights_path)
            except:
                print("Impossible to find weights for", model_name, "in ", weights_path)
                return

    return model, input_layer, output_layer, global_pool


def calculate_flops(model):
    """ Estimate flops for a standard ANN model"""
    connections = 0
    neurons = 0
    for layer in model.layers:
        print("-----", layer.name, "-----")
        if isinstance(layer, tf.keras.layers.InputLayer):
            print("\tNothing to do.\n")
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            w_shape = layer.kernel.shape
            stride = layer.strides
            if layer.padding == "valid":
                new_connections = (
                    np.floor((layer_in_shape[1] - w_shape[0] + stride[0]) / stride[0])
                    * np.floor((layer_in_shape[2] - w_shape[1] + stride[1]) / stride[1])
                    * w_shape[0]
                    * w_shape[1]
                    * w_shape[2]
                    * w_shape[3]
                )
            else:
                new_connections = (
                    np.floor((layer_in_shape[1] + stride[0] - 1) / stride[0])
                    * np.floor((layer_in_shape[2] - 1 + stride[1]) / stride[1])
                    * w_shape[0]
                    * w_shape[1]
                    * w_shape[2]
                    * w_shape[3]
                )
            new_neurons = layer_out_shape[1] * layer_out_shape[2] * layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections += new_connections
            neurons += new_neurons
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            w_shape = layer.pool_size
            stride = layer.strides
            if layer.padding == "valid":
                new_connections = (
                    np.floor((layer_in_shape[1] - w_shape[0] + stride[0]) / stride[0])
                    * np.floor((layer_in_shape[2] - w_shape[1] + stride[1]) / stride[1])
                    * w_shape[0]
                    * w_shape[1]
                    * layer_in_shape[3]
                )
            else:
                new_connections = (
                    np.floor((layer_in_shape[1] + stride[0] - 1) / stride[0])
                    * np.floor((layer_in_shape[2] - 1 + stride[1]) / stride[1])
                    * w_shape[0]
                    * w_shape[1]
                    * w_shape[2]
                    * layer_in_shape[3]
                )
            new_neurons = layer_out_shape[1] * layer_out_shape[2] * layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections += new_connections
            neurons += new_neurons
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            layer_in_shape = layer.input_shape
            new_connections = layer_in_shape[1] * layer_in_shape[2] * layer_in_shape[3]
            new_neurons = layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections += new_connections
            neurons += new_neurons
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            new_connections = layer_in_shape[1] * layer_out_shape[1]
            new_neurons = layer_out_shape[1]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections += new_connections
            neurons += new_neurons
        else:
            print("\tNothing to do.\n")

    print("Total connections:", connections)
    print("Total neurons:", neurons)
    flops = 2 * connections + neurons
    print("Estimated flops:", flops)
    return flops


def compute_connections(layer):
    """ Compute the average number of connections for neuron in a layer"""
    input_tensor = layer.input

    # Config of the layer
    config = layer.get_config()
    num_input_connections = 0

    if isinstance(layer, tf.keras.layers.InputLayer):
        num_input_connections = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        input_channels = (
            input_tensor.shape[-1]
            if config["data_format"] == "channels_last"
            else input_tensor.shape[1]
        )
        num_input_connections = np.prod(config["kernel_size"]) * input_channels

    elif isinstance(layer, tf.keras.layers.Dense):
        num_input_connections = input_tensor.shape[-1]

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        num_input_connections = np.prod(config["pool_size"])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        if config["data_format"] == "channels_last":
            num_input_connections = np.prod(input_tensor.shape[1:2])
        else:
            num_input_connections = np.prod(input_tensor.shape[-2:-1])
    # We assume that all other layer types are just passing the input
    else:
        num_input_connections = 0

    return num_input_connections


def compute_neurons(layer):
    """ Calculate the number of neurons in a layer"""

    neurons = 0
    if isinstance(layer, tf.keras.layers.InputLayer):
        neurons = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        neurons = np.prod(layer.output_shape[1:])

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        neurons = np.prod(layer.output_shape[1:])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        neurons = np.prod(layer.output_shape[1:])
    elif isinstance(layer, tf.keras.layers.Dense):
        neurons = layer.output_shape[1]
    else:
        print("\tNothing to do.\n")

    return neurons