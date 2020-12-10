# Nengo Space

## Table of contents
* [About this project](#about-this-project)
* [Requirements](#requirements)
* [Instalation](#installation)
* [Usage](#usage)
  * [Testing](#testing)
  * [Training a VGG16-based model on UC Merced dataset](#training-a-vgg16-based-model-on-uc-merced-dataset)
  * [Converting to a spiking neural network](#converting-to-a-spiking-neural-network)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About this project
This projects aims at testing the feasibility of using  [Nengo](https://www.nengo.ai/nengo/)-based spiking neural networks on Earth observation data, as well as other
space-related applications.

In particular, the current implementation converts a [VGG-16](https://neurohive.io/en/popular-networks/vgg16/)-based classifier, trained on
[UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html), into a [Nengo](https://www.nengo.ai/nengo/)-based spiking neural network, and
evaluates it on the same dataset.

## Requirements

This project uses [Python](https://www.python.org/) 3.6, and requires the following third party libraries:
* [NumPy](https://numpy.org/) 1.18.5
* [Matplotlib](https://matplotlib.org/) 3.3.3
* [TensorFlow](https://www.tensorflow.org/) 2.3.1
* [TensorFlow Datasets](https://www.tensorflow.org/datasets) 4.1.0
* [Nengo](https://www.nengo.ai/nengo/) 3.0.0
* [NengoDL](https://www.nengo.ai/nengo-dl/) 3.3.0

Different versions of Nengo packages will raise assertion errors. This is because there are compatibility issues between newer versions of
[Nengo](https://www.nengo.ai/nengo/) and the current [NengoDL](https://www.nengo.ai/nengo-dl/) (at the time or writing this). No issues related to using different versions of
the other libraries was encountered. Nonetheless, recommend creating a separate environment for this project and installing the versions of the packages specified above.
[NumPy](https://numpy.org/) can be acquired by installing the [Anaconda](https://www.anaconda.com/) distribution of [Python](https://www.python.org/).
Alternatively, all the required libraries can be installed using `pip`:

```pip install numpy==1.18.5 matplotlib==3.3.3 tensorflow==2.3.1 tensorflow-datasets==4.1.0 nengo==3.0.0 nengo-dl==3.3.0```

## Installation

To install this project either download it as a `.zip` file and extract it into a desired directory or clone it via the terminal or console command:
* using the HTTPS

```git clone https://github.com/AndrzejKucik/Nengo_Space.git```

* or SSH

```git clone git@github.com:AndrzejKucik/Nengo_Space.git```

## Usage

Change the current working directory to be the `Nengo_Space`directory.

### Testing

To test that the training of a [Keras](https://keras.io/)-based model and conversion to a [Nengo](https://www.nengo.ai/nengo/)-based spiking neural network works well, run:

```python nengo_test.py```

This will create a simple classifier model with two convolutional hidden layers and a single dense output layer. The layers do not use bias parameters (for biological
plausibility). This model is then trained for five epochs on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) with RMSprop optimizer and L<sub>2</sub> regularization
for the convolutional kernels. After the training and testing (it should achieve roughly 99% test set accuracy), the model is converted into a spiking neural network using
[NengoDL](https://www.nengo.ai/nengo-dl/) library. This network is then tested in the spiking settin, using 0.01 synapse and scaling the firing rate by 100. It should achieve
comparable test set performance in this setting. See the code for details.

### Training a VGG16-based model on UC Merced dataset

In the first par of the project we download the [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) (21 classes, 100 examples each). We
slice it into the training, validation, and test sets using ratios 80%-10%-10% (the data examples are ordered according to version
[2.0.0](https://www.tensorflow.org/datasets/catalog/uc_merced) in TensorFlow datasets) Because the dataset is very small, we agument thetraining set using random dihedral
group transformation, random crop, random brightness change, random contrast change, random hue change, random saturation change. We use a modified version of the
[VGG-16](https://neurohive.io/en/popular-networks/vgg16/) network trained on the [ImageNet](http://www.image-net.org/) dataset (parameters from the
[Keras-TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) version to construct a classifier for this dataset. We replace the max pooling
layers with average pooling layers in order to have a spiking neural network compatibility in [NengoDL](https://www.nengo.ai/nengo-dl/). We remove the head of the network
(all the layers following the last pooling layers) and replace it with a dense classifier layer. The final pooling layer before the classifier is a global, rather than local,
average pooling layer.
Optionally, one can add a dropout layer after each local average pooling layer.
To be compatible with the usual [VGG-16](https://neurohive.io/en/popular-networks/vgg16/) input, we resize the images to (224, 224, 3) shape.
The model is trained using the RMSprop optimzer, using early stopping and reducing the learning rate on plateau (by a factor of 10) if there is no significant improvement
in the validation loss after 100 and 50 consecutive epochs respectively.

To train the network run:

```
python vgg_ucm.py [-s seed] [-e epochs] [-bs batch_size] [-drpt dropout] [-kl2 kernel_l2] [-bl1 bias_l1] [-mbd max_brightness_delta]
[-mhd max_hue_delta] [-lc lower_contrast] [-uc upper contrast] [-ls lower_saturation] [-us upper saturation]

```

where the optional arguments are:
* `seed`
* `epochs`
* `batch_size`
* `dropout`
* `kernel_l2`
* `bl1 bias_l1`
* `max_brightness_delta`
* `max_hue_delta`
* `lower_contrast`
* `upper contrast`
* `lower_saturation`
* `upper saturation`

The default values of these parameters are the ones that empirically gave us the best test accuracy performance (91.43%). The original
[VGG-16](https://neurohive.io/en/popular-networks/vgg16/) model uses bias terms, which are not very biologically plausible, so we penalize them using L<sub>1</sub>
regularization. We also use  L<sub>2</sub> regularization for the convolutional kernel training to improve the spiking neural network performance later on.

The trained model is saved to 
```
./models/s_{seed}_e_{epochs}_bs_{batch_size}_drpt_{dropout}_kl2_{kernel_l2}_bl1_{bias_l1}_mbd_{max_brightness_delta}_
mhd_{max_hue_delta}_ls_{lower_contrast}_uc_{upper contrast}_ls_{lower_saturation}_us_{upper saturation}.h5
```

where each `{name}`is replace by the corresponding value of `name`.


**_Note that the obtained test set accuracy might be different each time (sometimes even failing to converge), even if identical parameters are used, because the global
random seed does not seem to affect the shuffling processes or data augmentation parameters._**

### Converting to a spiking neural network
A [VGG-16](https://neurohive.io/en/popular-networks/vgg16/)-based classifier trained on the
[UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) can be converted into a spiking neural network using
[NengoDL](https://www.nengo.ai/nengo-dl/) module by running

```
python vgg_ucm_nengo.py -md model_path [-sc firing_rate_scale} [-syn synapse] [-t timesteps]
```

where
* `model_path` is the path of a valid `.h5` model which we obtain after running `vgg_ucm.py`.
* `firing_rate_scale` is the scaling factor for the firing rate (float).
* `synapse` is the neurons' synapse value (float).
* `timesteps` is the number of timesteps of the simulation (int).

After the conversion and simulation five sample figures showing two input examples together with the spiking activity of the global pooling and the final dense layers plotted
against the timesteps will be saved to

```
./figs/{model_name}/scale_{scale}/synapse_{synapse}/timesteps_{timesteps}/acc_{accuracy}_{i}.png
```

where `{model_name}`is the name of
the model `.h5`file (without the extension), `{accuracy}`is its numerical accuracy on the test set, `{i}` is the data slice index, and the remaining parameters are as in the
item list above.

Generally, the spiking model accuracy is very sensitive to the parameters' choice. For example, running the model which achieves 91.43% accuracy in the frame-based context,
we get that the corresponding spiking neural network simulated for 200 time steps achieves the following level of accuracy (with respect to other parameters):

| **Synapse\Firing rate scale** |100     | 250    | 500    | 750    | 1000   |
|-------------------------------|--------|--------|--------|--------|--------|
| **0.001**                     | 18.10% | 46.19% | 76.19% | 82.85% | 85.23% |
| **0.005**                     | 63.24% | 85.23% | 89.52% | 90.47% | 90.95% |
| **0.01**                      | 59.52% | 84.76% | 89.52% | 90.00% | 90.00% |

## Roadmap
In the future, a spiking neural network initialised with the weights obtained by training a [VGG-16](https://neurohive.io/en/popular-networks/vgg16/)-based classfier on
[UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) will be further trained in the spiking context. For details on the progress of
spike-based training see: [SpykeTorch Space](https://github.com/GabrieleMeoni/SpykeTorch_Space).

## License
Property of the European Space Agency. Distributed under MIT licence. See [`LICENSE`](../master/LICENSE) for more information.

## Contact

E-mail: [andrzej.kucik@esa.int](mailto:andrzej.kucik@esa.int)

Project Link: <https://github.com/AndrzejKucik/Nengo_Space>

## Acknowledgements

[ESA-&Phi;-Lab](https://github.com/ESA-PhiLab)

[Gabriele Meoni](https://github.com/GabrieleMeoni/)

[TensorFlow](https://www.tensorflow.org/)

[Nengo](https://www.nengo.ai/)
