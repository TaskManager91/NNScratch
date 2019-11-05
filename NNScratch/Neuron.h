//	#####################################################################
//	#																	#
//	#	Neuron.h														#
//	#	contains logic for each neuron									#
//	#																	#
//	#####################################################################

#pragma once

#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
public:
	double a;
	int index;	// own position in layer
	double error_term;
	double mse;
	vector<double> weight;	// weights this neuron -> next layer 
	double sigmoid(double z);
	void feedForward(vector<Neuron> &lastLayer);
	void outputError(double y);
	void hiddenError(vector<Neuron> &nextLayer);
	Neuron(int weightCount, int myIndex);
};