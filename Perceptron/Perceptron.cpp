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

	for(int k = 0;k < number_neuron[i - 1];k++){
		weight[i][j][k] -= derivative[i][j] * neuron[i - 1][k];
	}
	weight[i][j][number_neuron[i - 1]] -= derivative[i][j];
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

	for(int k = 0;k < number_neuron[i - 1];k++){
		sum += neuron[i - 1][k] * weight[i][j][k];
	}
	neuron[i][j] = sum + weight[i][j][number_neuron[i - 1]];
}

Perceptron::Perceptron(int number_neuron[]){
	this->number_layer	= 2;
	this->number_neuron = new int[number_layer];

	for(int i = 0;i < number_layer;i++){
		this->number_neuron[i] = number_neuron[i];
	}

	derivative	= new double*[number_layer];
	neuron		= new double*[number_layer];

	for(int i = 0;i < number_layer;i++){
		derivative[i]	= new double[number_neuron[i]];
		neuron[i]		= new double[number_neuron[i]];
	}

	weight = new double**[number_layer];

	for(int i = 1;i < number_layer;i++){
		weight[i] = new double*[number_neuron[i]];

		for(int j = 0;j < number_neuron[i];j++){
			weight[i][j] = new double[number_neuron[i - 1] + 1];
		}
	}
}
Perceptron::~Perceptron(){
	for(int i = 0;i < number_layer;i++){
		delete[] derivative[i];
		delete[] neuron[i];
	}
	delete[] derivative;
	delete[] neuron;

	for(int i = 1;i < number_layer;i++){
		for(int j = 0;j < number_neuron[i];j++){
			delete[] weight[i][j];
		}
		delete[] weight[i];
	}
	delete[] weight;

	delete[] number_neuron;
}

void Perceptron::Initialize_Parameter(int seed, double scale, double shift){
	srand(seed);

	for(int i = 1;i < number_layer;i++){
		for(int j = 0;j < number_neuron[i];j++){
			for(int k = 0;k < number_neuron[i - 1] + 1;k++){
				weight[i][j][k] = scale * rand() / RAND_MAX + shift;
			}
		}
	}
}
void Perceptron::Test(double input[], double output[]){
	for(int j = 0;j < number_neuron[0];j++){
		neuron[0][j] = input[j];
	}
	for(int i = 1;i < number_layer;i++){
		for(int j = 0;j < number_neuron[i];j++){
			Feedforward	(i, j);
			Activate	(i, j);
		}
	}
	for(int j = 0;j < number_neuron[number_layer - 1];j++){
		output[j] = neuron[number_layer - 1][j];
	}
}

double Perceptron::Train(int number_training, double learning_rate, double **input, double **target_output){
	double loss = 0;

	for(int h = 0;h < number_training;h++){
		for(int j = 0;j < number_neuron[0];j++){
			neuron[0][j] = input[h][j];
		}
		for(int i = 1;i < number_layer;i++){
			for(int j = 0;j < number_neuron[i];j++){
				Feedforward	(i, j);
				Activate	(i, j);
			}
		}
		for(int i = number_layer - 1;i > 0;i--){
			for(int j = 0;j < number_neuron[i];j++){
				Differentiate	(i, j, learning_rate, target_output[h]);
				Adjust_Parameter(i, j);
			}
		}
		for(int j = 0;j < number_neuron[number_layer - 1];j++){
			loss += 0.5 * (neuron[number_layer - 1][j] - target_output[h][j]) * (neuron[number_layer - 1][j] - target_output[h][j]);
		}
	}
	return loss / number_training;
}