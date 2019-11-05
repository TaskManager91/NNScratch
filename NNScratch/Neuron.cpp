//	#####################################################################
//	#																	#
//	#	Neuron.cpp														#
//	#	contains logic for each neuron									#
//	#																	#
//	#####################################################################

#include "Neuron.h"

Neuron::Neuron(int weightCount, int myIndex)
{
	for (int i = 0; i < weightCount; i++)
		weight.push_back(rand() / double(RAND_MAX));
		
	index = myIndex;
}

void Neuron::feedForward(vector<Neuron> &lastLayer)
{
	double z = 0.0;

	for (int neuron = 0; neuron < lastLayer.size(); neuron++)
		z += lastLayer[neuron].a * lastLayer[neuron].weight[index];

	// sigmoid func
	a = sigmoid(z);
}

double Neuron::sigmoid(double z) {
	double a = (1 / (1 + exp(-z)));
	return a;
}

void Neuron::outputError(double y)
{
	// (y - a) * sigmoid func derivative
	error_term = - (y - a) * (a * (1 - a));
	mse = (a - y) * (a - y);
}

void Neuron::hiddenError(vector<Neuron> &nextLayer)
{
	double sum = 0.0;

	for (int neuron = 0; neuron < nextLayer.size() - 1; neuron++)
		sum += weight[neuron] * nextLayer[neuron].error_term;

	// SUM( w * error_term[n+1] ) * sigmoid func derivative
	error_term = sum * (a*(1-a));
}
