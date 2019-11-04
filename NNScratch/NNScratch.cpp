// NNScratch.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <fstream>
#include <locale>
#include "NeuralNet.h"

int interpolate(double x);

int main()
{
	std::locale::global(std::locale("de-DE"));
	std::cout << "NNScratch! by Christoph B. \n";

	vector<int> structure = { 8,3,8 };

	vector<vector<double>> data{	{1,0,0,0,0,0,0,0},
									{0,1,0,0,0,0,0,0},
									{0,0,1,0,0,0,0,0},
									{0,0,0,1,0,0,0,0},
									{0,0,0,0,1,0,0,0},
									{0,0,0,0,0,1,0,0},
									{0,0,0,0,0,0,1,0},
									{0,0,0,0,0,0,0,1} };

	int inputCounter = 0;

	double lambda = 0.00000001;    // learning Rate
	double alpha = 0.95;	// alpha

	NeuralNet sarah(structure, alpha, lambda);

	for (int i = 0; i < sarah.network.size(); i++)
		cout << "Layer : " << i << " Neurons: " << sarah.network[i].size() << endl;

	cout << endl;

	unsigned epochs = 2000;
	vector<double> result, training_input, training_output;
	double mse = 0.0;
	double mseBuffer = 0.0;
	bool trained = false;

	ofstream myfile;
	myfile.open("data.csv");
	myfile << "Epoch;MSE;ERROR;Alpha: " << alpha << ";Lambda: "<< lambda << "\n";

	for (int i = 0; i <= epochs; i++)
	{
		for(int j = 0; j <= data.size(); j++)
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

			cout << "RAW: ";
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