# NNScratch

A from scratch neural network that uses the sigmoid function.

As an example a 3 layer net was implemented (input-hidden-output) that learns how to compress 9 inputs over a 3 neuron hidden layer.

# Quickstart

[NNScratch.cpp](https://github.com/TaskManager91/NNScratch/blob/master/NNScratch/NNScratch.cpp "NNScratch.cpp") contains the example implementation with the configuration of the neural network.  

Alpha/Lambda will be initialized as double, epochs as unsigned.

The structure of the network itself is defined by a vector: 
```cpp
// structure of the Neural Net 8 Input Neurons 3 hidden Neurons 8 Ouput Neurons
vector<int> structure = { 8,3,8 };
```

The main logic of the network can be found in [NeuralNet.cpp](https://github.com/TaskManager91/NNScratch/blob/master/NNScratch/NeuralNet.cpp "NeuralNet.cpp").

The [Neuron.cpp](https://github.com/TaskManager91/NNScratch/blob/master/NNScratch/Neuron.cpp "Neuron.cpp") forms the individual neurons and contains the sigmoid function.
