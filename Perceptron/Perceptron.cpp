#include <math.h>
#include <stdlib.h>

#include "Perceptron.h"

void Perceptron::Activate(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	neuron[i][j] = 1 / (1 + exp(-neuron[i][j]));
}
void Perceptron::Adjust_Parameter(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	for(int k = 0;k < number_neurons[i - 1];k++){
		weight[i][j][k] -= derivative[i][j] * neuron[i - 1][k];
	}
	weight[i][j][number_neurons[i - 1]] -= derivative[i][j];
}
void Perceptron::Differentiate(int layer_index, int neuron_index, double learning_rate, double target_output[]){
	int i = layer_index;
	int j = neuron_index;

	derivative[i][j] = learning_rate * (neuron[i][j] - target_output[j]) * neuron[i][j] * (1 - neuron[i][j]);
}
void Perceptron::Feedforward(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double sum = 0;

	for(int k = 0;k < number_neurons[i - 1];k++){
		sum += neuron[i - 1][k] * weight[i][j][k];
	}
	neuron[i][j] = sum + weight[i][j][number_neurons[i - 1]];
}

Perceptron::Perceptron(int number_neurons[]){
	this->number_layers	= 2;
	this->number_neurons = new int[number_layers];

	for(int i = 0;i < number_layers;i++){
		this->number_neurons[i] = number_neurons[i];
	}

	derivative	= new double*[number_layers];
	neuron		= new double*[number_layers];

	for(int i = 0;i < number_layers;i++){
		derivative[i]	= new double[number_neurons[i]];
		neuron[i]		= new double[number_neurons[i]];
	}

	weight = new double**[number_layers];

	for(int i = 1;i < number_layers;i++){
		weight[i] = new double*[number_neurons[i]];

		for(int j = 0;j < number_neurons[i];j++){
			weight[i][j] = new double[number_neurons[i - 1] + 1];
		}
	}
}
Perceptron::~Perceptron(){
	for(int i = 0;i < number_layers;i++){
		delete[] derivative[i];
		delete[] neuron[i];
	}
	delete[] derivative;
	delete[] neuron;

	for(int i = 1;i < number_layers;i++){
		for(int j = 0;j < number_neurons[i];j++){
			delete[] weight[i][j];
		}
		delete[] weight[i];
	}
	delete[] weight;

	delete[] number_neurons;
}

void Perceptron::Initialize_Parameter(int seed, double scale, double shift){
	srand(seed);

	for(int i = 1;i < number_layers;i++){
		for(int j = 0;j < number_neurons[i];j++){
			for(int k = 0;k < number_neurons[i - 1] + 1;k++){
				weight[i][j][k] = scale * rand() / RAND_MAX + shift;
			}
		}
	}
}
void Perceptron::Test(double input[], double output[]){
	for(int j = 0;j < number_neurons[0];j++){
		neuron[0][j] = input[j];
	}
	for(int i = 1;i < number_layers;i++){
		for(int j = 0;j < number_neurons[i];j++){
			Feedforward	(i, j);
			Activate	(i, j);
		}
	}
	for(int j = 0;j < number_neurons[number_layers - 1];j++){
		output[j] = neuron[number_layers - 1][j];
	}
}

double Perceptron::Train(int number_training, double learning_rate, double **input, double **target_output){
	double loss = 0;

	for(int h = 0;h < number_training;h++){
		for(int j = 0;j < number_neurons[0];j++){
			neuron[0][j] = input[h][j];
		}
		for(int i = 1;i < number_layers;i++){
			for(int j = 0;j < number_neurons[i];j++){
				Feedforward	(i, j);
				Activate	(i, j);
			}
		}
		for(int i = number_layers - 1;i > 0;i--){
			for(int j = 0;j < number_neurons[i];j++){
				Differentiate	(i, j, learning_rate, target_output[h]);
				Adjust_Parameter(i, j);
			}
		}
		for(int j = 0;j < number_neurons[number_layers - 1];j++){
			loss += 0.5 * (neuron[number_layers - 1][j] - target_output[h][j]) * (neuron[number_layers - 1][j] - target_output[h][j]);
		}
	}
	return loss / number_training;
}
