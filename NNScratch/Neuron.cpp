#include "Neuron.h"

double Neuron::lambda = 0.0001;    // learning Rate
double Neuron::alpha = 0.7;   // alpha

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
	a = (1 / (1 + exp(-z)));
}

void Neuron::outputError(double y)
{
	// (y - a) * sigmoid func derivative
	error_term = - (y - a) * (a * (1 - a));
}

void Neuron::hiddenError(vector<Neuron> &nextLayer)
{
	double sum = 0.0;

	for (int neuron = 0; neuron < nextLayer.size() - 1; neuron++)
		sum += weight[neuron] * nextLayer[neuron].error_term;

	// SUM( w * error_term[n+1] ) * sigmoid func derivative
	error_term = sum * (a*(1-a));
}

void Neuron::updateWeight(vector<Neuron> &prevLayer)
{
	for (int neuronCount = 0; neuronCount < prevLayer.size(); neuronCount++) 
	{
		Neuron &neuron = prevLayer[neuronCount]; 

		//new Weight = oldWeight - alpha [ (error_term(l+1) * activation) + (lambda * oldweight) ]
		neuron.weight[index] = neuron.weight[index] - alpha * ((error_term * neuron.a) + (lambda * neuron.weight[index]));
	}
}

