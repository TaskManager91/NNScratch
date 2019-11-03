#include "NeuralNet.h"

vector<double> NeuralNet::getOutput()
{
	vector<double> outputLayer;
	for (int i = 0; i < network.back().size() - 1; i++) 
		outputLayer.push_back(network.back()[i].a);

	return outputLayer;
}

NeuralNet::NeuralNet(vector<int> structure)
{
	int layerCount = structure.size();

	for (int layer = 0; layer < layerCount; layer++) 
	{
		vector<Neuron> newLayer;
		network.push_back(newLayer);

		int weightCount = layer == layerCount - 1 ? 0 : structure[layer + 1];

		for (int neuron = 0; neuron <= structure[layer]; neuron++) 
		{
			network[layer].push_back(Neuron(weightCount, neuron));
			
			if (neuron == structure[layer])
				network[layer][neuron].a = 1.0;
		}
	}
}

void NeuralNet::feedForward(vector<double> inputLayer)
{
	// Set input neurons
	for (int neuron = 0; neuron < inputLayer.size(); neuron++)
		network[0][neuron].a = inputLayer[neuron];

	// feed forward 
	for (int layer = 1; layer < network.size(); layer++) 
	{
		vector<Neuron> lastLayer = network[layer - 1];

		for (int neuron = 0; neuron < network[layer].size() - 1; neuron++) 
			network[layer][neuron].feedForward(lastLayer);
	}
}

void NeuralNet::backPropagation(vector<double> &target)
{
	vector<Neuron> &outputLayer = network.back();

	// output layer error terms
	for (int neuron = 0; neuron < outputLayer.size() - 1; neuron++) 
		outputLayer[neuron].outputError(target[neuron]);

	// hidden layer error terms
	for (int layerCount = network.size() - 2; layerCount > 0; layerCount--) {
		vector<Neuron> &layer = network[layerCount];
		vector<Neuron> &nextLayer = network[layerCount + 1];

		for (int neuron = 0; neuron < layer.size(); neuron++) 
			layer[neuron].hiddenError(nextLayer);
	}

	// update connection weights
	for (int layerCount = network.size() - 1; layerCount > 0; layerCount--) {
		vector<Neuron> &layer = network[layerCount];
		vector<Neuron> &lastLayer = network[layerCount - 1];

		for (int neuron = 0; neuron < layer.size() - 1; neuron++) {
			layer[neuron].updateWeight(lastLayer);
		}
	}
}
