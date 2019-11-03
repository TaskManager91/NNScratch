// NNScratch.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include "NeuralNet.h"

int interpolate(double x);

int main()
{
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

	NeuralNet sarah(structure);

	for (int i = 0; i < sarah.network.size(); i++)
		cout << "Layer : " << i << " Neurons: " << sarah.network[i].size() << endl;

	cout << endl;

	unsigned epochs = 20000;
	vector<double> result, training_input, training_output;

	for (unsigned i = 0; i <= epochs; i++)
	{
		training_input = data[inputCounter];
		sarah.feedForward(training_input);

		result = sarah.getOutput();

		training_output = data[inputCounter];

		sarah.backPropagation(training_output);

		inputCounter++;

		if (inputCounter == data.size())
			inputCounter = 0;

		if (i % 1000 == 0 || i >= (epochs - 20))
		{
			cout << "Iteration: " << i << endl;

			cout << " Input: ";
			for (unsigned i = 0; i < training_input.size(); i++)
				cout << training_input[i] << ' ';
			cout << '\n';

			cout << "Result: ";
			for (unsigned i = 0; i < result.size(); i++)
				cout << interpolate(result[i]) << ' ';
			cout << '\n';

			cout << "Output: ";
			for (unsigned i = 0; i < training_output.size(); i++)
				cout << training_output[i] << ' ';
			cout << '\n';

			cout << endl;
		}
	}
}

int interpolate(double x)
{
	if (x >= 0.5)
		return 1;
	else
		return 0;
}