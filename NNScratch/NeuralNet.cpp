#include "NeuralNet.h"


NeuralNet::NeuralNet(vector<int> structure, double aBuffer, double lBuffer)
{
	int layerCount = structure.size();

	for (int layer = 0; layer < layerCount; layer++)
	{
		vector<Neuron> newLayer;
		network.push_back(newLayer);

		// If last layer output =0, else number of elements from next layer
		int outputs = layer == layerCount - 1 ? 0 : structure[layer + 1];

		for (int neuron = 0; neuron <= structure[layer]; neuron++)
		{
			network[layer].push_back(Neuron(outputs, neuron));

			//Set Bias
			if ((neuron == structure[layer]) && (layer != layerCount))
				network[layer][neuron].a = 1.0;
		}
	}

	alpha = aBuffer;
	lambda = lBuffer;
}

double NeuralNet::getMSE()
{
	double mse = 0.0;
	for (int i = 0; i < network.back().size() - 1; i++)
		mse += network.back()[i].mse;

	return mse;
}


vector<double> NeuralNet::getOutput()
{
	vector<double> outputLayer;
	for (int i = 0; i < network.back().size() - 1; i++) 
		outputLayer.push_back(network.back()[i].a);

	return outputLayer;
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
			Neuron &bufferNeuron = layer[neuron];

			for (int neuronLast = 0; neuronLast < lastLayer.size()-1; neuronLast++)
			{
				Neuron &neuron = lastLayer[neuronLast];

				//new Weight = oldWeight - alpha [ (error_term(l+1) * activation) + (lambda * oldweight) ]
				neuron.weight[bufferNeuron.index] = neuron.weight[bufferNeuron.index] - alpha * ((bufferNeuron.error_term * neuron.a) + (lambda * neuron.weight[bufferNeuron.index]));
			}
			Neuron &biasNeuron = lastLayer[lastLayer.size()-1];
			biasNeuron.weight[bufferNeuron.index] = biasNeuron.weight[bufferNeuron.index] - alpha * (bufferNeuron.error_term * biasNeuron.a);
		}
	}
}
