#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuralNet
{
public:
	double lambda;					// learning Rate
	double alpha;					// 
	vector<vector<Neuron>> network; // network[layer][neuron]
	vector<double> getOutput();
	double getMSE();
	void feedForward(vector<double> inputVals);
	void backPropagation(vector<double>& targetVals);
	NeuralNet(vector<int> structure, double alpha, double lambda);
};

