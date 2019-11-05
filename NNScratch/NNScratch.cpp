//	#####################################################################
//	#																	#
//	#	NNScratch														#
//	#	by Christoph Bensch & Oliver Bensch								#
//	#																	#
//	#																	#
//	#	Building a neural network from scratch							#
//	#																	#
//	#																	#
//	#	MAIN (NNScratch.cpp)											#
//	#																	#
//	#####################################################################

#include <string>
#include <iostream>
#include <fstream>
#include <locale>
#include <iomanip> 
#include "NeuralNet.h"

using namespace std;

int interpolate(double x);
void consoleLog(int epoch, NeuralNet network);

//	--- CONFIG --
double lambda = 0.000001;	// learning Rate
double alpha = 4.5;			// alpha
unsigned epochs = 20000;	// number of iterations
bool interpolated = false;

// structure of the Neural Net 8 Input Neurons 3 hidden Neurons 8 Ouput Neurons
vector<int> structure = { 8,3,8 };

//	------------------------------------ MAIN ------------------------------------
int main()
{
	// Sets the double values from 0.0 to 0,0 for cout, so that Excel can read the data later.
	locale::global(locale("de-DE"));

	cout << "NNScratch! by Christoph B. & Oliver B. \n";

	//	Load config
	double inputlambda, inputalpha;
	unsigned inputepochs;
	char inputinterpolated;
	cout << "Show interpolated results (1=true, 0=false default:1): ";
	cin >> inputinterpolated;

	if (inputinterpolated == '0')
		interpolated = false;
	else
		interpolated = true;

	// Create NeuralNet object
	NeuralNet sarah(structure, alpha, lambda);		// sarah, the AI from the eureka series.

	//	Print out number of neurons for each layer
	for (int i = 0; i < sarah.network.size(); i++)
		cout << "Layer : " << i << " Neurons: " << sarah.network[i].size() << endl;

	// Init necessary variables for the net
	vector<double> result, training_input, training_output;
	double mse = 0.0;
	double mseBuffer = 0.0;
	bool correct = true;

	// logs everything into a CSV so it can be visualized later with Excel
	ofstream myfile;
    myfile.open("data.csv");
	myfile << "Epoch ;MSE ;Correct ;Alpha: " << alpha << ";Lambda: "<< lambda << "\n";

	// training set
	vector<vector<double>> data = { {1,0,0,0,0,0,0,0},
									{0,1,0,0,0,0,0,0},
									{0,0,1,0,0,0,0,0},
									{0,0,0,1,0,0,0,0},
									{0,0,0,0,1,0,0,0},
									{0,0,0,0,0,1,0,0},
									{0,0,0,0,0,0,1,0},
									{0,0,0,0,0,0,0,1} };
	int inputCounter = 0;

	// ---------------------------------- MAIN  LOOP ----------------------------------	
	for (int i = 0; i <= epochs; i++)
	{
		training_input = data[inputCounter];

		//	----- Feedforward -----
		sarah.feedForward(training_input);

		result = sarah.getOutput();

		training_output = data[inputCounter];

		//	----- backpropagation ----- 
		sarah.backPropagation(training_output);

		correct = true;
		for (int i = 0; i < result.size(); i++){
			if (interpolate(result[i]) != training_output[i])
				correct = false;
		}

		mseBuffer += sarah.getMSE();

		inputCounter++;

		if (i % 1000 == 0)
			consoleLog(i, sarah);

		// Log to CSV
		if (i % 10 == 0 && i > 20)
			myfile << i << ";" << mse << ";" << correct << "\n";

		if (inputCounter == data.size())
		{
			mse = mseBuffer / data.size();
			mseBuffer = 0;
			inputCounter = 0;
		}
	}

	myfile.close();

	char test;
	cout << "Press enter to exit.\n";
	cin >> test;
}

void consoleLog(int epoch, NeuralNet network) {
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

int interpolate(double x)
{
	if (x >= 0.5)
		return 1;
	else
		return 0;
}