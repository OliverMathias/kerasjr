<img src="https://i.imgur.com/64tEH5H.png"  align= right height="200" width="200" />

# kerasjr
A python package to create and train simple dense neural networks.

## Table of contents
* [What is it?](#what-is-it?)
* [Methods Overview](#methods-overview)
* [How to Import](#how-to-import)
* [Size Comparison](#size-comparison)
* [Examples](#examples)
* [License](#license)

## What is it?
kerasjr is a python package that allows users to create simple dense neural networks with minimal setup or code. Our aim was to create a package that was a fraction of the size of Keras but worked well for smaller and simpler datasets that only require dense neural networks.

Below is a quick slide comparing the strengths & weaknesses of kerasjr and Keras.

![](https://i.imgur.com/azCy98X.png)

## Methods Overview
kerasjr allows users to pick between 3 different layer activations, 3 loss functions, the number of hidden layers and the number of hidden nodes in each layer.

The default code for kerasjr uses tanh activation functions for each layer and allows users to get a network up and running in lines. It looks like this...
``` 
model = Model(x, y, number_of_hidden_layers=1, number_of_hidden_nodes=10)
model.train("mse", 10000, alpha=.001)
```
In the first line we are initializing the model and passing in the x and y data for training, the number of hidden layers and the number of nodes in each hidden layer.
``` python
model = Model(x, y, number_of_hidden_layers=1, number_of_hidden_nodes=10)
```

In the next line we simple pass in the activation function, the number of epochs, and the alpha or "learning rate".
``` python
model.train("mse", 10000, alpha=.001)
```

Below are code examples of how to modify each feature...

### Picking Activation Functions
If we wanted to modify the activation functions of any layer we could do so easily as shown below...

Modifying the input layer's activation to sigmoid...
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.input_layer_activation_function = "sigmoid" <<<<<
model.train("cce", 10, alpha=.001)
```

Modifying all the hidden layers' activation functions to leaky_relu
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.hidden_layer_activation_function = "leaky_relu" <<<<<
model.train("cce", 10, alpha=.001)
```

Modifying the output layer's activation function to tanh
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.output_layer_activation_function = "tanh" <<<<<
model.train("cce", 10, alpha=.001)
```
**Note**, all these changes can be done to any layer interchangeably.


### Picking The Loss Function
If we wanted to modify the loss function of the network we could do so easily as shown below...

Modifying the networks loss function to mean squared error...
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.output_layer_activation_function = "tanh"
model.train("mse", 10, alpha=.001) <<<<<
```

Modifying the networks loss function to mean absolute error...
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.output_layer_activation_function = "tanh"
model.train("mae", 10, alpha=.001) <<<<<
```

Modifying the networks loss function to categorical cross entropy...
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.output_layer_activation_function = "tanh"
model.train("cce", 10, alpha=.001) <<<<<
```

### Picking The Number of Hidden Layers and Their Nodes
Picking the number of hidden using and nodes in each unit is really simple. The code below is an example of the instantiation of our model, in which we tell it to create a dense network with a single hidden layer and 10 nodes.
```
model = Model(x, y, number_of_hidden_layers=1, number_of_hidden_nodes=10)
```
When we run the code above we get an output from kerasjr that details our network architecture.
```
Network Architecture:
----------------------------------------------------------------------------
Input Layer Number of Weights: 30
Hidden Layer 1 Number of Weights: 100
Output Layer Number of Weights: 10
----------------------------------------------------------------------------
Total Number of Weights:  140
```
The number of weights in the input layer will be the shape of the input data X the number of nodes in the hidden layer(s). The number of weights in the hidden layer(s) will be the `(number of nodes * 2 ) * number of layers` because each node needs to connect to each of the next nodes.

Below is the network architecture diagram from the example above...

![](https://i.imgur.com/2Eb9dHH.jpg)


### Using Predict
The final method in kerasjr is the `predict` method. This is just a forward pass through the network using the trained weights to do interence on passed in data. Below is an example of a how to use the `predict` method, simple instantiate a model, train it, and call `predict` on some data
``` python
model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
model.hidden_layer_activation_function = "leaky_relu"
model.train("cce", 10, alpha=.001)
model.predict(x)

```

## How to Import
Importing kerasjr is as simple as running a pip install command...
```
pip install kerasjr
```
It's also just as easy to clone our github repo...
```
git clone https://github.com/OliverMathias/kerasjr.git
```

## Size Comparison
kerasjr's only real dependency is numpy, which makes up +99% of it's download size. Numpy is used for the matrix multiplication within kerasjr and is integral to it's functioning. Even with the numpy included in our size calculation, as shown below, kerasjr is almost 80% smaller than Keras.

Which means you'll be able to set up a dense neural network a lot faster.

![](https://i.imgur.com/FMoRVLf.png)

It's also helpful to see the raw size of each component written out, as in the bar chart above, kerasjr's base package isn't even visible because of how comparatively small it is.
Here we can see all the dependencies of each package and their sizes written in kilobytes.

![](https://i.imgur.com/t3jNood.png)
## Examples
Below are a few examples to show how to use this package for common regression problems.
To run these examples, please follow the steps outlined in "How to Import", and execute the two import lines below...
``` python
from kerasjr import Model
import numpy as np
```

  * ### Example 1 (Simple Toy Dataset)
  ``` python
  #inputs
  x = np.array([[0,0,0], [1,1,1], [1,1,1], [0,0,0]])
  #output
  y = np.array([[0],[1],[1],[0]])

  model = Model(x, y, number_of_hidden_layers=1, number_of_hidden_nodes=10)
  model.train("mse", 10000, alpha=.001)
  ```
  Output:
  ```
  Network Architecture:

  Input Layer Number of Weights: 30
  Hidden Layer 1 Number of Weights: 100
  Output Layer Number of Weights: 10

  Total Number of Weights:  140

  Epoch: 1 ERROR: 0.9289321392754415
  Epoch: 1001 ERROR: 0.04779347601709738
  Epoch: 2001 ERROR: 0.02916774269680078
  Epoch: 3001 ERROR: 0.021612982110729373
  Epoch: 4001 ERROR: 0.017348308568584825
  Epoch: 5001 ERROR: 0.014552223342823245
  Epoch: 6001 ERROR: 0.01255299217368111
  Epoch: 7001 ERROR: 0.011040118930825349
  Epoch: 8001 ERROR: 0.009848617798073556
  Epoch: 9001 ERROR: 0.008882018237726275

  Done.
  Final Accuracy: 99.19194531361383%
  ```
  * ### Example 2 (Wheat Seed Length)
  ``` python
  dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

  x = []
  y = []
  for i in dataset:
    x.append(i[:2])
    y.append(i[2])

  x = np.asanyarray(x)
  y = np.asanyarray([y])

  model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
  model.hidden_layer_activation_function = "leaky_relu"
  model.train("cce", 1500, alpha=.001)
  ```
  Output:
  ```
  Network Architecture:

  Input Layer Number of Weights: 100
  Hidden Layer 1 Number of Weights: 2500
  Hidden Layer 2 Number of Weights: 2500
  Output Layer Number of Weights: 500

  Total Number of Weights:  5600

  Epoch: 1 ERROR: 39.44551846880063
  Epoch: 151 ERROR: 4.9949999667039416e-09
  Epoch: 301 ERROR: 4.9949999667039416e-09
  Epoch: 451 ERROR: 4.9949999667039416e-09
  Epoch: 601 ERROR: 4.9949999667039416e-09
  Epoch: 751 ERROR: 4.9949999667039416e-09
  Epoch: 901 ERROR: 4.9949999667039416e-09
  Epoch: 1051 ERROR: 4.9949999667039416e-09
  Epoch: 1201 ERROR: 4.9949999667039416e-09
  Epoch: 1351 ERROR: 4.9949999667039416e-09

  Done.
  Final Accuracy: 99.9999995005%
  ```
  * ### Example 3 (Gas Mileage Prediction)
  ``` python
  from sklearn.datasets import load_boston
  boston = load_boston()
  x = boston["data"]
  y = boston["target"]
  y = y.reshape((x.shape[0], 1))

  model = Model(x, y, number_of_hidden_layers=2, number_of_hidden_nodes=50)
  model.hidden_layer_activation_function = "leaky_relu"
  model.train("cce", 10, alpha=.001)
  ```
  Output:
  ```
  Network Architecture:

  Input Layer Number of Weights: 650
  Hidden Layer 1 Number of Weights: 2500
  Hidden Layer 2 Number of Weights: 2500
  Output Layer Number of Weights: 50

  Total Number of Weights:  5700

  Epoch: 1 ERROR: 171.98800780793812
  Epoch: 2 ERROR: 2.2510273367735832e-08
  Epoch: 3 ERROR: 2.2510273367735832e-08
  Epoch: 4 ERROR: 2.2510273367735832e-08
  Epoch: 5 ERROR: 2.2510273367735832e-08
  Epoch: 6 ERROR: 2.2510273367735832e-08
  Epoch: 7 ERROR: 2.2510273367735832e-08
  Epoch: 8 ERROR: 2.2510273367735832e-08
  Epoch: 9 ERROR: 2.2510273367735832e-08
  Epoch: 10 ERROR: 2.2510273367735832e-08

  Done.
  Final Accuracy: 99.99999774897266%
  ```

## License
MIT License: Check the License file for full permissions.

