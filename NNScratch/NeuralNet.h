//	#####################################################################
//	#																	#
//	#	NeuralNet.cpp													#
//	#	contains logic for the whole neural network						#
//	#																	#
//	#####################################################################

#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"

using namespace std;

class NeuralNet
{
public:
	double lambda;					
	double alpha;					
	vector<vector<Neuron>> network;		// network[layer][neuron]
	vector<double> getOutput();			// returns the last layer (output layer)
	double getMSE();					// returns the MSE (output Layer)
	void feedForward(vector<double> inputVals);
	void backPropagation(vector<double>& targetVals);
	NeuralNet(vector<int> structure, double alpha, double lambda);
};

