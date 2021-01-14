import tensorflow as tf
import numpy as np
from alexnet import AlexNet

def create_model(STRATEGY,
                 model_name="vgg16",
                 input_shape=(224,224,3),
                 num_classes=21,
                 kernel_l2=1e-4,
                 bias_l1=1e-5,
                 dropout=0,
                 lr=0.01):
    """Creates a Keras model which is a modified version of the VGG16 network."""

    if model_name == "vgg16":
        # Load the VGG16 model (this may take a moment for the first time)
        original_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
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

        filters, kernel_size, name, padding = config['filters'], config['kernel_size'], config['name'], config['padding']
        # -- We keep the same number of filters and the kernel size but change the activation format and add
        # -- kernel and bias regularizers
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
                                   bias_regularizer=tf.keras.regularizers.l1(bias_l1),
                                   name=name)(input_layer)

        # - The remaining layers
        for layer in original_model.layers[2:-1]:
            try:
                config = layer.get_config()
                filters, kernel_size, name, padding = config['filters'], config['kernel_size'], config['name'], config['padding']
                x = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           activation=tf.nn.relu,
                                           kernel_regularizer=tf.keras.regularizers.l2(kernel_l2),
                                           bias_regularizer=tf.keras.regularizers.l1(bias_l1),
                                           name=name)(x)
            except KeyError:
                # -- Change the max pooling to average pooling
                x = tf.keras.layers.AveragePooling2D((2, 2))(x)
                # -- Add dropout if necessary
                if dropout > 0.:
                    x = tf.keras.layers.Dropout(dropout)(x)

        # - Conclude VGG layers with a global average pooling layer
        global_pool = tf.keras.layers.GlobalAveragePooling2D()(x)

        # - Output layer
        output_layer = tf.keras.layers.Dense(num_classes, use_bias=False)(global_pool)

        # - Define the model
        model = tf.keras.Model(input_layer, output_layer)

        # - After the model is defined, we can load the weights
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = original_model.get_layer(name=layer.name).get_weights()
                layer.set_weights(weights)

        # - Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.metrics.SparseCategoricalAccuracy())

    return model


def calculate_flops(model):
    """ Estimate flops for a standard ANN model"""
    connections = 0
    neurons = 0
    for layer in model.layers:
        print("-----",layer.name,"-----")
        if isinstance(layer, tf.keras.layers.InputLayer):
            print("\tNothing to do.\n")
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            w_shape = layer.kernel.shape
            stride = layer.strides
            if layer.padding == "valid":
                new_connections=np.floor((layer_in_shape[1] - w_shape[0] + stride[0])/stride[0]) * np.floor((layer_in_shape[2] - w_shape[1] + stride[1])/stride[1]) * w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
            else:
                new_connections=np.floor((layer_in_shape[1] + stride[0] - 1)/stride[0]) * np.floor((layer_in_shape[2] -1 + stride[1])/stride[1]) * w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
            new_neurons=layer_out_shape[1]*layer_out_shape[2]*layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections+=new_connections
            neurons+=new_neurons
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            w_shape = layer.pool_size
            stride = layer.strides
            if layer.padding == "valid":
                new_connections=np.floor((layer_in_shape[1] - w_shape[0] + stride[0])/stride[0]) * np.floor((layer_in_shape[2] - w_shape[1] + stride[1])/stride[1]) * w_shape[0] * w_shape[1] * layer_in_shape[3]
            else:
                new_connections=np.floor((layer_in_shape[1] + stride[0] - 1)/stride[0]) * np.floor((layer_in_shape[2] -1 + stride[1])/stride[1]) * w_shape[0] * w_shape[1] * w_shape[2] * layer_in_shape[3]
            new_neurons=layer_out_shape[1]*layer_out_shape[2]*layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections+=new_connections
            neurons+=new_neurons
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            layer_in_shape = layer.input_shape
            new_connections=layer_in_shape[1] * layer_in_shape[2] * layer_in_shape[3]
            new_neurons=layer_out_shape[3]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections+=new_connections
            neurons+=new_neurons
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_in_shape = layer.input_shape
            layer_out_shape = layer.output_shape
            new_connections=layer_in_shape[1]*layer_out_shape[1]
            new_neurons=layer_out_shape[1]
            print("\t Found", new_connections, "connections.")
            print("\t Found", new_neurons, "neurons.\n")
            connections+=new_connections
            neurons+=new_neurons
        else:
            print("\tNothing to do.\n")
    
    print("Total connections:", connections)
    print("Total neurons:", neurons)
    flops = 2 * connections + neurons
    print("Estimated flops:", flops)
    return flops