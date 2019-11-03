#pragma once

#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(int outputs, int index);
	double a;
	void feedForward(vector<Neuron> &lastLayer);
	void outputError(double targetVal);
	void hiddenError(vector<Neuron> &nextLayer);
	void updateWeight(vector<Neuron> &prevLayer);;
private:
	static double lambda; // learning Rate
	static double alpha; // alpha
	vector<double> weight;	// weights this neuron -> next layer 
	int index;	// own position in layer
	double error_term;
};