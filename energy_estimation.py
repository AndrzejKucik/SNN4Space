#!/usr/bin/env python3.6

"""Estimators of flops, MACs, ACs, and energy consumption of ANN and SNN models during the inference.."""

# -- Built-in modules -- #
from argparse import ArgumentParser

# -- Third-party modules -- #
import numpy as np
import tensorflow as tf

# -- Proprietary modules -- #
from dataloaders import load_eurosat, load_ucm
from utils import rescale_resize_image, INPUT_FILTER_DICT

import nengo
import nengo_dl
import numpy as np
from time import time
from datetime import timedelta


# -- File info -- #
__author__ = ["Andrzej S. Kucik", "Gabriele Meoni"]
__copyright__ = "European Space Agency"
__contact__ = "andrzej.kucik@esa.int"
__version__ = "0.1.2"
__date__ = "2021-02-12"

color_dictionary = {
    "red": "\033[0;31m",
    "black": "\033[0m",
    "green": "\033[0;32m",
    "orange": "\033[0;33m",
    "purple": "\033[0;35m",
    "blue": "\033[0;34m",
    "cyan": "\033[0;36m",
}


def compute_connections(layer):
    """Compute the average number of connections for neuron in a layer.
    Parameters
        ----------
        layer : tf.keras.layer
            Input keras layet.

    Returns
        -------
            num_input_connections : int
                Number of average number connections per neuron."""
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
    """Calculate the number of neurons in a layer.
    Parameters
        ----------
        layer : tf.keras.layer
            Input keras layet.

    Returns
    -------
        neurons : int
            Number of neurons in layer."""

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


def energy_estimate_layer(
    model,
    layer_idx,
    spikes_measurements=None,
    probe_layers=None,
    dt=0.001,
    spiking_model=True,
    device="cpu",
    if_neurons=False,
    verbose=False,
):
    """Estimate the energy spent for synpatic operations and neurons update required for an inference for an Artificial or Spiking layer on a target hardware.
    Energy is estimated by multiplying the number of synaptic operations and neuron updates times the energy values for a single synpatic operation and neuron update.
    Energy values for synpatic operation and neuron update are obtained by KerasSpiking: https://www.nengo.ai/keras-spiking/.
    Parameters
        ----------
        model : tf.keras.Model
            Keras model.
        layer_idx : int
            Layer index.
        spikes_measurements: SimulationData (Default=None)
            Simulation data produced during the simuation of the Nengo model. It can be None for Artificial Neural Networks.
        probe_layers: list (Default=None)
            List of probes for the different layers of the Nengo model.
        dt: int (Default=0.001)
            Nengo simulator time resolution
        spiking_model: bool (Default=True)
            Flag to indicate if the model is a spiking model or not
        device: str (Default="cpu")
            Device name of the target hardware. Supported "cpu" (Intel i7-4960X),  "gpu" (Nvidia GTX Titan Black), "arm" (ARM Cortex-A), "loihi", "spinnaker", "spinnaker2".
        if_neurons: bool (Default=False)
            If True, Integrate and Fire (IF) neurons are modelled.  Such neurons can be updated only when an input spike is provided (if the hardware supports sporadic activations).
            If False, Leaky Integrate and Fire neurons are modelled, which are updated once per timestep. (Note: use this even for IF neurons with synaptic filters).
        verbose: bool (Default=False)
            If True, energy contributions are shown for every single layer.
        Returns
        -------
            synop_energy  : int
                Energy contribution for synaptic operations.
            neuron_energy : int
                Energy contribution for neuron updates."""
    # Energy costants for the different operations
    devices = {
        # https://ieeexplore.ieee.org/abstract/document/7054508
        "cpu": dict(spiking=False, energy_per_synop=8.6e-9, energy_per_neuron=8.6e-9),
        "gpu": dict(spiking=False, energy_per_synop=0.3e-9, energy_per_neuron=0.3e-9),
        "arm": dict(spiking=False, energy_per_synop=0.9e-9, energy_per_neuron=0.9e-9),
        # https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
        "loihi": dict(
            spiking=True,
            energy_per_synop=(23.6 + 3.5) * 1e-12,
            energy_per_neuron=81e-12,
        ),
        # https://arxiv.org/abs/1903.08941
        "spinnaker": dict(
            spiking=True, energy_per_synop=13.3e-9, energy_per_neuron=26e-9
        ),
        "spinnaker2": dict(
            spiking=True, energy_per_synop=450e-12, energy_per_neuron=2.19e-9
        ),
    }
    energy_dict = devices[device]
    if (spiking_model == True) and (energy_dict["spiking"] == False):
        print("Error! Impossible to infer Spiking models on standard hardware!")
        return -1.0, -1.0

    # Energy per synaptic operation
    energy_per_synop = energy_dict["energy_per_synop"]
    # Energy per neuron update
    energy_per_neuron = energy_dict["energy_per_neuron"]
    # Number of neurons in a layer
    n_neurons = compute_neurons(model.layers[layer_idx])
    # Average number of input connections per neuron
    n_connections_per_neuron = compute_connections(model.layers[layer_idx])
    n_timesteps = 0
    if spiking_model == False:
        f_in = 1 / dt
        n_timesteps = 1
    else:
        if layer_idx == 0:
            f_in = 0
        else:
            spikes_in = spikes_measurements[probe_layers[layer_idx - 1]]
            n_timesteps = spikes_in.shape[1]
            f_in = np.sum(spikes_in) / (
                spikes_in.shape[0] * n_timesteps * spikes_in.shape[2] * dt
            )

    # synop_energy/inference = energy/op * ops/event * events/s * s/timestep * timesteps/inference
    synop_energy = (
        energy_per_synop
        * n_connections_per_neuron
        * n_neurons
        * f_in
        * dt
        * n_timesteps
    )
    if if_neurons:  # neurons are update only when an output spike is produced
        # neuron_energy/inference = energy/op * ops/event * events/s * s/timestep * timesteps/inference
        neuron_energy = energy_per_neuron * n_neurons * f_in * dt * n_timesteps
    else:  # neurons are update every timestep (for instance, to implement alpha functions)
        # neuron_energy/inference = energy/op * ops/timestep * timesteps/inference
        neuron_energy = energy_per_neuron * n_neurons * n_timesteps
    if verbose:
        print("--- Layer: ", model.layers[layer_idx].name, " ---")
        print("\tSynop energy: ", synop_energy, "J/inference")
        print("\tNeuron energy: ", neuron_energy, "J/inference")
        print("\tTotal energy: ", neuron_energy + synop_energy, "J/inference")
    return synop_energy, neuron_energy


# noinspection PyUnboundLocalVariable
def main():
    """The main function."""
    # - Argument parser - #
    parser = ArgumentParser()
    parser.add_argument(
        "-md",
        "--model_path",
        type=str,
        default="",
        required=True,
        help="Path to the model.",
    )
    parser.add_argument(
        "-if",
        "--input_filter",
        type=str,
        default="",
        help="Type of the input filter (if any).",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=1,
        help="Number of timesteps for the spiking model simulation.",
    )
    parser.add_argument(
        "-d_a",
        "--device_ann",
        type=str,
        default="arm",
        help='Target device for energy estimation for the ANN model. Supported "cpu" (Intel i7-4960X),  "gpu" (Nvidia GTX Titan Black), "arm" (ARM Cortex-A), "loihi", "spinnaker", "spinnaker2". ',
    )
    parser.add_argument(
        "-d_s",
        "--device_snn",
        type=str,
        default="loihi",
        help='Target device for energy estimation for the SNN model. Supported "loihi", "spinnaker", "spinnaker2". ',
    )
    parser.add_argument(
        "-sc", "--scale", type=float, default=1, help="Post simulation scale value."
    )

    parser.add_argument(
        "-v", "--verbose", action='store_true', default=False, help="If True, the energy contributions for all the layers is shown."
    )

    parser.add_argument(
        "-n",
        "--n_test",
        type=int,
        default=1,
        help="Number of images used for energy estimation.",
    )

    parser.add_argument(
        "-sy",
        "--synapse",
        type=float,
        default=0.1,
        help="Synaptic filter alpha constant.",
    )

    args = vars(parser.parse_args())
    path_to_model = args["model_path"]
    input_filter = args["input_filter"].lower()
    timesteps = args["timesteps"]
    device_ann = args["device_ann"]
    device_snn = args["device_snn"]
    scale = args["scale"]
    n_test = args["n_test"]
    synapse = args["synapse"]
    verbose = args["verbose"]

    # Load model
    try:
        model = tf.keras.models.load_model(filepath=path_to_model)
        model.summary()
    except OSError:
        exit("Invalid model path!")

    # Input and output shapes
    input_shape = model.input.shape[1:]
    num_classes = model.output.shape[-1]

    if input_shape == (64, 64, 3) and num_classes == 10:
        print(
            "Using",
            color_dictionary["red"],
            "EuroSAT",
            color_dictionary["black"],
            "dataset...",
        )
        _, _, x_test, labels = load_eurosat()
        num_test = 2700
    elif input_shape == (224, 224, 3) and num_classes == 21:
        print(
            "Using",
            color_dictionary["red"],
            "UCM",
            color_dictionary["black"],
            "dataset...",
        )
        _, _, x_test, labels = load_ucm()
        num_test = 210
    else:
        exit("Invalid model!")

    # Preprocessing function
    def rescale_resize(image, label):
        """Rescales and resizes the input images."""
        return rescale_resize_image(image, input_shape[:-1]), label

    # noinspection PyUnboundLocalVariable
    x_test = x_test.map(
        rescale_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(batch_size=num_test)
    
    if input_filter in INPUT_FILTER_DICT.keys():
        x_test = x_test.map(
            INPUT_FILTER_DICT[input_filter],
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    print(
        color_dictionary["green"],
        "Converting model to Nengo model...",
        color_dictionary["black"],
    )
    # Convert to a Nengo network
    converter = nengo_dl.Converter(
        model,
        scale_firing_rates=scale,
        synapse=synapse,
        swap_activations={tf.nn.relu: nengo.SpikingRectifiedLinear()},
    )

    network_input = converter.inputs[model.input]
    network_output = converter.outputs[model.output]
    n_layer = len(converter.layers)  # Number of layers
    probe_layer = []
    with converter.net:
        for layer in model.layers:
            probe_layer.append(nengo.Probe(converter.layers[layer]))
        nengo_dl.configure_settings(stateful=False)

    
    x_test = np.array([n[0] for n in x_test]).squeeze(0)
    y_test = np.array([n[1] for n in x_test])

    x_test = x_test[0:n_test]

    # Tile images according to the number of timesteps
    tiled_test_images = np.tile(
        np.reshape(x_test, (x_test.shape[0], 1, -1)), (1, timesteps, 1)
    )
    test_labels = y_test.reshape((y_test.shape[0], 1, -1))

    print(color_dictionary["blue"], "Start simulation...", color_dictionary["black"])

    with nengo_dl.Simulator(converter.net) as sim:
        # Record how much time it takes
        start = time()
        data = sim.predict({network_input: tiled_test_images})
        print(
            "Time to make a prediction with {} timestep(s): {}.".format(
                timesteps, timedelta(seconds=time() - start)
            )
        )

    print(color_dictionary["cyan"],"--------- ANN model ---------",color_dictionary["black"])
    print("\tHardware: ", color_dictionary["red"], device_ann, color_dictionary["black"])
    neuron_energy, synop_energy = 0, 0
    for layer_idx in range(len(model.layers)):
        neuron_energy_layer, synop_energy_layer = energy_estimate_layer(
            model,
            layer_idx,
            spikes_measurements=data,
            probe_layers=probe_layer,
            dt=sim.dt,
            spiking_model=False,
            device=device_ann,
            verbose=verbose,
        )
        neuron_energy += neuron_energy_layer
        synop_energy += synop_energy_layer
    print(color_dictionary["orange"],"\t--------- Total energy ---------", color_dictionary["black"])
    print("\tSynop energy: ", synop_energy, "J/inference")
    print("\tNeuron energy: ", neuron_energy, "J/inference")
    print("\tTotal energy:",color_dictionary["green"], synop_energy + neuron_energy, color_dictionary["black"],"J/inference\n\n")

    neuron_energy, synop_energy = 0, 0
    print(color_dictionary["cyan"],"--------- SNN model ---------",color_dictionary["black"])
    print("\tHardware: ", color_dictionary["red"], device_snn, color_dictionary["black"])
    for layer_idx in range(len(model.layers)):
        neuron_energy_layer, synop_energy_layer = energy_estimate_layer(
            model,
            layer_idx,
            spikes_measurements=data,
            probe_layers=probe_layer,
            dt=sim.dt,
            spiking_model=True,
            device=device_snn,
            verbose=verbose,
        )
        neuron_energy += neuron_energy_layer
        synop_energy += synop_energy_layer
    print(color_dictionary["orange"],"\t--------- Total energy ---------", color_dictionary["black"])
    print("\tSynop energy: ", synop_energy, "J/inference")
    print("\tNeuron energy: ", neuron_energy, "J/inference")
    print("\tTotal energy:",color_dictionary["green"], synop_energy + neuron_energy, color_dictionary["black"],"J/inference\n\n")


if __name__ == "__main__":
    main()
