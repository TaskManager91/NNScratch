// NNScratch.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <fstream>
#include <locale>
#include <iomanip> 
#include "NeuralNet.h"

int interpolate(double x);
void output(int epoch, NeuralNet network);
void outputFinal(NeuralNet network);

//Config
double lambda = 0.0001;		// learning Rate
double alpha = 0.95;		// alpha
unsigned epochs = 20000;	// number of iterations
bool interpolated = false;

vector<int> structure = { 8,3,8 };

int main()
{
	locale::global(locale("de-DE"));
	cout << "NNScratch! by Christoph B. & Oliver B. \n";

	//load config
	double inputlambda, inputalpha, inputepochs;
	char inputinterpolated;
	/*cout << "Select an lambda value (default 0.0001): ";
	cin >> inputlambda;
	cout << "\n Select an alpha value (default 0.95): ";
	cin >> inputalpha;
	cout << "\n Select the number of iterations (default 20000): ";
	cin >> inputepochs;*/
	cout << "\n Show interpolated results (1=true, 0=false default:1): ";
	cin >> inputinterpolated;

	if (inputinterpolated == '0')
		interpolated = false;
	else
		interpolated = true;

	cout << "n";

	// Create NeuralNet object
	NeuralNet sarah(structure, alpha, lambda);

	//Print out number of neurons for each layer
	for (int i = 0; i < sarah.network.size(); i++)
		cout << "Layer : " << i << " Neurons: " << sarah.network[i].size() << endl;
	cout << endl;

	//
	vector<double> result, training_input, training_output;
	double mse = 0.0;
	double mseBuffer = 0.0;
	bool trained = false;

	ofstream myfile;
	myfile.open("data.csv");
	myfile << "Epoch;MSE;ERROR;Alpha: " << alpha << ";Lambda: "<< lambda << "\n";

	vector<vector<double>> data = { {1,0,0,0,0,0,0,0},
							{0,1,0,0,0,0,0,0},
							{0,0,1,0,0,0,0,0},
							{0,0,0,1,0,0,0,0},
							{0,0,0,0,1,0,0,0},
							{0,0,0,0,0,1,0,0},
							{0,0,0,0,0,0,1,0},
							{0,0,0,0,0,0,0,1} };

	int inputCounter = 0;
	for (int i = 0; i <= epochs; i++)
	{
		training_input = data[inputCounter];
		sarah.feedForward(training_input);

		result = sarah.getOutput();

		training_output = data[inputCounter];

		sarah.backPropagation(training_output);

		for (int i = 0; i < result.size(); i++)
		{
			if (interpolate(result[i]) != training_output[i])
				trained = true;
		}

		mseBuffer += sarah.getMSE();

		inputCounter++;
			
		if (inputCounter == data.size())
		{
			mse = mseBuffer / data.size();
			trained = false;
			mseBuffer = 0;
			inputCounter = 0;
		}

		if (i % 1000 == 0)
			output(i, sarah);

		if (i % 100 == 0 && i != 0)
			myfile << i << ";" << mse << ";" << trained << "\n";
	}

	//outputFinal(sarah);

	myfile.close();

	char test;
	cout << "Press enter to exit.\n";
	cin >> test;
}

void output(int epoch, NeuralNet network) {
	cout << "Iteration: " << epoch << endl;

	vector<vector<double>> data = { {1,0,0,0,0,0,0,0},
							{0,1,0,0,0,0,0,0},
							{0,0,1,0,0,0,0,0},
							{0,0,0,1,0,0,0,0},
							{0,0,0,0,1,0,0,0},
							{0,0,0,0,0,1,0,0},
							{0,0,0,0,0,0,1,0},
							{0,0,0,0,0,0,0,1} };

	cout << "Output: \n";
	for (int j = 0; j <= data.size() - 1; j++)
	{
		vector<double> input = data[j];
		network.feedForward(input);

		vector<double> result = network.getOutput();

		for (int i = 0; i < result.size(); i++) {
			if(interpolated)
				cout << interpolate(result[i]);
			else
				cout << fixed << setprecision(3) << result[i] << ' ';
		}

		cout << " Hidden Layer: ";
		for (int i = 0; i < network.network[1].size()-1; i++) {
			if(interpolated)
				cout << interpolate(network.network[1][i].a) << ' ';
			else
				cout << fixed << setprecision(3) << network.network[1][i].a << ' ';
		}

		cout << '\n';
	}
	double error = network.getMSE();
	cout << "MSE: " << error << "\n";
}

void outputFinal(NeuralNet network) {
	cout << "Weight matrix input to hidden: \n";
	for (int i = 0; i < network.network[0].size() - 1; i++) {
		cout << "Layer: " << i << ' ';
		for (int j = 0; j < network.network[0][i].weight.size(); j++) {
			cout << fixed << setprecision(2) << network.network[0][i].weight[j] << ' ';
		}
		cout << '\n';
	}

	cout << "Weight matrix hidden to output: \n";
	for (int i = 0; i < network.network[1].size() - 1; i++) {
		cout << "Layer: " << i << ' ';
		for (int j = 0; j < network.network[1][i].weight.size(); j++) {
			cout << fixed << setprecision(2) << network.network[1][i].weight[j] << ' ';
		}
		cout << '\n';
	}
}

int interpolate(double x)
{
	if (x >= 0.5)
		return 1;
	else
		return 0;
}