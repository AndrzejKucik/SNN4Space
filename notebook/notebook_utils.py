import keras_spiking
import tensorflow as tf
import numpy as np
from utils import COLOUR_DICTIONARY


def compute_connections(layer):
    """
    Compute the average number of connections for neuron in a layer.

    Parameters
    ----------
    layer : tf.keras.layer
        Keras layer.

    Returns
     -------
    num_input_connections : int
        Number of average number connections per neuron.
    """

    # Input tensor
    input_tensor = layer.input

    # Config of the layer
    config = layer.get_config()

    if isinstance(layer, tf.keras.layers.InputLayer):
        num_input_connections = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        input_channels = (input_tensor.shape[-1] if config['data_format'] == 'channels_last' else input_tensor.shape[1])
        num_input_connections = np.prod(config['kernel_size']) * input_channels

    elif isinstance(layer, tf.keras.layers.Dense):
        num_input_connections = input_tensor.shape[-1]

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        num_input_connections = np.prod(config['pool_size'])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        if config['data_format'] == 'channels_last':
            num_input_connections = np.prod(input_tensor.shape[1:2])
        else:
            num_input_connections = np.prod(input_tensor.shape[-2:-1])

    # We assume that all other layer types are just passing the input
    else:
        num_input_connections = 0

    return num_input_connections


def compute_neurons(layer):
    """
    Calculate the number of neurons in a layer.
    Parameters
    ----------
    layer : tf.keras.layer
        Keras layer.

    Returns
    -------
    neurons : int
        Number of neurons in the layer.
    """

    neurons = 0
    if isinstance(layer, tf.keras.layers.InputLayer):
        neurons = 0
    elif isinstance(layer, tf.keras.layers.Conv2D):
        neurons = np.prod(layer.output_shape[2:])

    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        neurons = np.prod(layer.output_shape[2:])

    elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        neurons = np.prod(layer.output_shape[2:])
    elif isinstance(layer, tf.keras.layers.Dense):
        neurons = layer.output_shape[1]
    else:
        neurons=0

    return neurons

def extract_model_info(model, verbose=False):  
    """
    Providing a list of number of input connections per neuron, number of neurons for Conv2D, Dense, and GlobalAveragePool2D layers, and providing the indices of the layers for which the activations shall be extracted, including LowPass filters, Dense and GlobalAveragePooling activations.
    ----------
    layer : tf.keras.Model
        Keras Model.
    verbose: boolean (default: False)
        If True, verbose logging is provided.

    Returns
    -------
    number_of_connection_per_neuron_list : list
        List of the number of input connection per neuron for the different layers.
    number_of_neurons_list : list
        List of the number of neurons for the different layers.
    activations_to_track_index_list : list
        List containing the indeces of the layers for which activations shall be extracted.
        
    """  
    
    number_of_connection_per_neuron_list=[]
    number_of_neurons_list=[]
    number_of_activations_to_track = 0
    number_of_connection_info_to_track = 0
    activations_to_track_index_list = []
    layer_index=0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) or isinstance(layer, tf.keras.layers.Dense):
            number_of_connection_info_to_track+=1
            
            if verbose:
                print("Found new info layer to track: ",number_of_connection_info_to_track)
                print("Layer name", layer.name)
            number_of_connection_per_neuron_list.append(compute_connections(layer))
            number_of_neurons_list.append(compute_neurons(layer))
        
        
        if isinstance(layer, keras_spiking.layers.Lowpass) or isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) or isinstance(layer, tf.keras.layers.Dense):
            number_of_activations_to_track+=1
            activations_to_track_index_list.append(layer_index)
            if verbose:
                print("Found new activation to track: ",number_of_activations_to_track)
                print("Layer name", layer.name)
        
        layer_index+=1
    return number_of_connection_per_neuron_list, number_of_neurons_list,activations_to_track_index_list


def extract_dataset(dataset_path, train_percentage=0.8, test_percentage=0.15, batch_size=1, img_size=(64, 64)):
    """
    Extracting and splitting a target dataset according to the train, test, validation percentages. The dataset is formatted according to the target batch size, and the images properly reshaped in the (x,y) plane.

    ----------
    dataset_path : string 
        Dataset path
    
    train_percentage: float (default: 0.8)
        Percentage of training data
    test_percentage: float (default: 0.15)
        Percentage of test data. Valid data percentage = 1 - (train_percentage + test_percentage)
    batch_size: int (default: 1)
        Batch size
    img_size: tuple (int, int)
        Image size.

    Returns
    -------
    train_ds : tf.raw_ops.BatchDataset 
        Training dataset split.
    valid_ds : tf.raw_ops.BatchDataset 
        Validation dataset split.
    test_ds : tf.raw_ops.BatchDataset 
        Test dataset split.
    """


    print("Extracting training data...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_path,
                validation_split=(1-train_percentage),
                subset="training",
                seed=123,
                image_size=img_size,
                batch_size=batch_size)

    print("Extracting validation data...")
    valid_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_path,
                validation_split=(1-train_percentage),
                subset="validation",
                seed=123,
                image_size=img_size,
                batch_size=batch_size)
    
    num_train_images=0
    num_train_batches=0
    for x,y in train_ds:
        num_train_images+=len(x)
        num_train_batches+=1

    num_valid_test_batches=0
    for x,y in valid_test_ds:
        num_valid_test_batches+=1

    
    valid_percentage = 1 -(train_percentage + test_percentage)
    total_batches=num_train_batches+num_valid_test_batches
    num_valid_batches=np.floor(valid_percentage*total_batches)
    valid_ds = valid_test_ds.take(num_valid_batches)
    test_ds = valid_test_ds.skip(num_valid_batches)
    return [train_ds, valid_ds, test_ds]


devices = {
        # https://ieeexplore.ieee.org/abstract/document/7054508
        'cpu': dict(spiking=False, energy_per_synop=8.6e-9, energy_per_neuron=8.6e-9),
        'gpu': dict(spiking=False, energy_per_synop=0.3e-9, energy_per_neuron=0.3e-9),
        'arm': dict(spiking=False, energy_per_synop=0.9e-9, energy_per_neuron=0.9e-9),
        'myriad2': dict(spiking=False, energy_per_synop=1.9918386085474344e-10,
                        energy_per_neuron=1.9918386085474344e-10),

        # Value estimated by considering the energy for a MAC operation. Such energy (E_per_mac) is obtained through a
        # maximum-likelihood estimation: E_inf = E_per_mac * N_ops by, E_inf and N_ops values come from our previous
        # work: https://ieeexplore.ieee.org/abstract/document/8644728

        # https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
        'loihi': dict(spiking=True, energy_per_synop=(23.6 + 3.5) * 1e-12, energy_per_neuron=81e-12),

        # https://arxiv.org/abs/1903.08941
        'spinnaker': dict(spiking=True, energy_per_synop=13.3e-9, energy_per_neuron=26e-9),
        'spinnaker2': dict(spiking=True, energy_per_synop=450e-12, energy_per_neuron=2.19e-9),
    } #Device parmeters per energy estimation


def energy_estimation(model, x_test_t=None, spiking_model=True, device_list=["loihi"], n_timesteps=10, dt=0.01, verbose=False):
    """
    Estimate the energy spent for synaptic operations and neurons update required for an inference for an Artificial or
    Spiking layer on a target hardware list. Energy is estimated by multiplying the number of synaptic operations and neuron
    updates times the energy values for a single synaptic operation and neuron update. The number of synaptic operations is obtained
    by performing the inference on a test_dataset for a spiking model. Energy values for synaptic
    operation and neuron update are obtained by KerasSpiking: https://www.nengo.ai/keras-spiking/. Energy values for Myriad 2 devices is obtained by mean square error interpolation
    of the values provided in: Benelli, Gionata, Gabriele Meoni, and Luca Fanucci. "A low power keyword spotting algorithm for memory constrained embedded systems." 2018 IFIP/IEEE International Conference on Very Large Scale Integration (VLSI-SoC). IEEE, 2018.
    Parameters
    ----------
    model : tf.keras.Model
        Keras model.
    x_test_t: tf.raw_ops.ParallelMapDataset (default=None)
        Simulation dataset to extract number of synaptic operations for a spiking model. It can be None for Artificial Neural
        Networks.
    spiking_model: bool (default=True)
        Flag to indicate if the model is a spiking model or not.
    device_list: list (default=['loihi'])
        List containing the name of the target hardware devices. Supported `cpu` (Intel i7-4960X), `gpu` (Nvidia GTX Titan Black), `arm` (ARM
        Cortex-A), `loihi`, `spinnaker`, `spinnaker2`, `myriad2`.
    n_timesteps: int (default=10)
        Number of simulation timesteps. For artificial models, n_timesteps is ignored and 1 is used.
    dt: float (Default=0.001)
        Nengo simulator time resolution

    
    verbose: bool (default=False)
        If `True`, energy contributions are shown for every single layer and additional log info is provided.
    Returns
    -------
    synop_energy_list : list
        List including the energy contribution for synaptic operations for each target device.
    neuron_energy_list : list
         List including the energy contribution for neuron updates for each target device.
    """

    print("Exctracting model info...")
    [number_of_connection_per_neuron_list, number_of_neurons_list,activations_to_track_index_list]=extract_model_info(model, verbose)
    
    if spiking_model:
        print("Found a spiking model.")
        activations=np.zeros([len(activations_to_track_index_list)])
        n_data=0
        print("Exctracting intermediate activations...")
        for x_t, y_t in x_test_t:
            n_data+=1
            n_layer=0
            for n in activations_to_track_index_list:
                new_model = tf.keras.Model(model.input, model.get_layer(index = 3).output)
                new_model.compile()
                activation = new_model.predict(x_t)
                activations[n_layer]+=tf.math.reduce_mean(tf.abs(activation))
                n_layer+=1
                
        activations/=n_data
    else:
        print("Found a not-Spiking model.")
        activations = np.ones([len(number_of_connection_per_neuron_list)])*1/dt
        n_timesteps = 1
        
    synop_energy_list=[]
    neuron_energy_list=[]
    for device in device_list:
        energy_dict = devices[device]
        if spiking_model and not energy_dict['spiking']:
            print(COLOUR_DICTIONARY['red'],'Error!', COLOUR_DICTIONARY['purple'],'Impossible to infer Spiking models on standard hardware!',COLOUR_DICTIONARY['black'])
            break

        # Energy per synaptic operation
        energy_per_synop = energy_dict['energy_per_synop']

        # Energy per neuron update
        energy_per_neuron_update = energy_dict['energy_per_neuron']
        print("Estimating energy on ", COLOUR_DICTIONARY['red'], device, COLOUR_DICTIONARY['black'])
        synop_energy=0
        neuron_energy=0

        for n in range(len(activations_to_track_index_list)):
            n_connections_per_neuron=number_of_connection_per_neuron_list[n]
            n_neurons=number_of_neurons_list[n]
            synop_energy_layer=energy_per_synop*n_connections_per_neuron*n_neurons*activations[n]*n_timesteps*dt
            neuron_energy_layer = energy_per_neuron_update * n_neurons * n_timesteps
            if verbose:
                print(COLOUR_DICTIONARY['blue'], "\t\tInspecting layer", model.layers[activations_to_track_index_list[n]], COLOUR_DICTIONARY['black'])
                print(COLOUR_DICTIONARY['orange'], '\t\t\t--------- Total energy ---------', COLOUR_DICTIONARY['black'])
                print('\t\t\t\tSynop layer energy: ', synop_energy_layer, 'J/inference')
                print('\t\t\t\tNeuron layer energy: ', neuron_energy_layer, 'J/inference')
                print('\t\t\tTotal layer energy:', COLOUR_DICTIONARY['green'], synop_energy + neuron_energy, COLOUR_DICTIONARY['black'],'J/inference\n\n')
    
            synop_energy+=synop_energy_layer
            neuron_energy+=neuron_energy_layer
        
        print(COLOUR_DICTIONARY['red'], "Global model energy", COLOUR_DICTIONARY['black'])
        print(COLOUR_DICTIONARY['orange'], '\t--------- Total energy ---------', COLOUR_DICTIONARY['black'])
        print('\tSynop layer energy: ', synop_energy, 'J/inference')
        print('\tNeuron layer energy: ', neuron_energy, 'J/inference')
        print('\tTotal layer energy:', COLOUR_DICTIONARY['green'], synop_energy + neuron_energy, COLOUR_DICTIONARY['black'],'J/inference\n\n')
        
        synop_energy_list.append(synop_energy)
        neuron_energy_list.append(neuron_energy)
    return synop_energy_list, neuron_energy_list
