#include "Neuron.h"

double Neuron::lRate = 0.1;    // learning Rate
double Neuron::alpha = 0.2;   //alpha

Neuron::Neuron(int outputs, int myIndex)
{
	for (int a = 0; a < outputs; a++)
	{
		weight.push_back(rand() / double(RAND_MAX));
		deltaWeight.push_back(0.5);
	}
		
	index = myIndex;
}

void Neuron::feedForward(vector<Neuron> &lastLayer)
{
	double z = 0.0;

	for (int neuron = 0; neuron < lastLayer.size(); neuron++)
		z += lastLayer[neuron].a * lastLayer[neuron].weight[index];

	//sigmoid func
	a = (1 / (1 + exp(-z)));
}

void Neuron::outputError(double y)
{
	// (y - a) * sigmoid func derivative
	error_term = (y - a) * (a * (1 - a));
}

void Neuron::hiddenError(vector<Neuron>& nextLayer)
{
	double sum = 0.0;

	for (int neuron = 0; neuron < nextLayer.size() - 1; neuron++)
		sum += weight[neuron] * nextLayer[neuron].error_term;

	// SUM( w * error_term[n+1] ) * sigmoid func derivative
	error_term = sum * (a * (1 - a));
}



void Neuron::updateWeight(vector<Neuron> &prevLayer)
{
	for (int neuron = 0; neuron < prevLayer.size(); neuron++) {
		Neuron &bufferNeuron = prevLayer[neuron];
		double oldDeltaWeight = bufferNeuron.deltaWeight[index];

		double newDeltaWeight =	lRate * bufferNeuron.a * error_term	+ alpha * oldDeltaWeight;

		bufferNeuron.deltaWeight[index] = newDeltaWeight;
		bufferNeuron.weight[index] += newDeltaWeight;
	}
}