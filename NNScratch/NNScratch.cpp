// NNScratch.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <fstream>
#include <locale>
#include "NeuralNet.h"

int interpolate(double x);

//Config
double lambda = 0.0001;		// learning Rate
double alpha = 0.95;		// alpha
unsigned epochs = 1000;		// number of iterations

vector<int> structure = { 8,3,8 };

vector<vector<double>> data{ {1,0,0,0,0,0,0,0},
							{0,1,0,0,0,0,0,0},
							{0,0,1,0,0,0,0,0},
							{0,0,0,1,0,0,0,0},
							{0,0,0,0,1,0,0,0},
							{0,0,0,0,0,1,0,0},
							{0,0,0,0,0,0,1,0},
							{0,0,0,0,0,0,0,1} };

int main()
{
	std::locale::global(std::locale("de-DE"));
	std::cout << "NNScratch! by Christoph B. \n";

	// Create NeuralNet object
	NeuralNet sarah(structure, alpha, lambda);

	//Print out number of neurons for each layer
	for (int i = 0; i < sarah.network.size(); i++)
		cout << "Layer : " << i << " Neurons: " << sarah.network[i].size() << endl;
	cout << endl;

	vector<double> result, training_input, training_output;
	double mse = 0.0;
	double mseBuffer = 0.0;
	bool trained = false;

	ofstream myfile;
	myfile.open("data.csv");
	myfile << "Epoch;MSE;ERROR;Alpha: " << alpha << ";Lambda: "<< lambda << "\n";

	int inputCounter = 0;
	for (int i = 0; i <= epochs; i++)
	{
		for(int j = 0; j <= 10; j++)
		{
			training_input = data[inputCounter];
			sarah.feedForward(training_input);

			result = sarah.getOutput();

			training_output = data[inputCounter];

			sarah.backPropagation(training_output);
		}

		for (int i = 0; i < result.size(); i++)
		{
			if (interpolate(result[i]) != training_output[i])
				trained = true;
		}

		mseBuffer += sarah.getMSE();

		inputCounter++;

		// write to file and ignore the "initial" rounds
		if(i >= 32)
			myfile << i << ";" << mse << ";" << trained << "\n";

		if (inputCounter == data.size())
		{
			mse = mseBuffer / data.size();
			trained = false;
			mseBuffer = 0;
			inputCounter = 0;
		}
			
		if (i % 1000 == 0 || i >= (epochs - 8))
		{
			cout << "Iteration: " << i << endl;

			cout << "Hidden Layer activation: ";
			for (int i = 0; i < sarah.network[1].size(); i++)
				cout << sarah.network[1][i].a << ' ';
			cout << '\n';

			cout << "RAW Output: ";
			for (int i = 0; i < result.size(); i++)
				cout << result[i] << ' ';
			cout << '\n';

			cout << " Input: ";
			for (int i = 0; i < training_input.size(); i++)
				cout << training_input[i] << ' ';
			cout << '\n';

			cout << "Result: ";
			for (int i = 0; i < result.size(); i++)
				cout << interpolate(result[i]) << ' ';
			cout << '\n';

			cout << "Output: ";
			for (int i = 0; i < training_output.size(); i++)
				cout << training_output[i] << ' ';
			cout << '\n';

			cout << "MSE: "<< mse << '\n';

			cout << endl;
		}
	}

	myfile.close();
}

int interpolate(double x)
{
	if (x >= 0.5)
		return 1;
	else
		return 0;
}