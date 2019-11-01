#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuralNet
{
public:
	NeuralNet(vector<int> structure);
	void feedForward(vector<double> inputVals);
	void backPropagation(vector<double>& targetVals);
	vector<double> getOutput();
	vector<vector<Neuron>> network; // network[layer][neuron]
};

