#include <fstream>
#include <iostream>

#include "MLP.h"

void Multilayer_Perceptron::Activate(char option[], int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	if (type_layer[i][0] == 'C'){
		if (strstr(type_layer[i], "bn")){
			Batch_Normalization_Activate(option, i, j);
		}

		for (int h = 0; h < batch_size; h++){
			double &neuron = this->neuron[0][i][h][j];

			if (strstr(type_layer[i], "ht")){
				neuron = 2 / (1 + exp(-2 * neuron)) - 1;
			}
			else
			if (strstr(type_layer[i], "ls")){
				neuron = 1 / (1 + exp(-neuron));
			}
			else{
				neuron *= (neuron > 0);
			}

			// Dropout *************************************************
			if (strstr(type_layer[i], "do")){
				char *rate = strstr(type_layer[i], "do") + 2;

				if (!strcmp(option, "train")){
					#pragma omp atomic
					neuron *= ((double)rand() / RAND_MAX <= atof(rate));
				}
				else
				if (!strcmp(option, "test")){
					neuron *= atof(rate);
				}
			}
			// *********************************************************
		}
	}
	else
	if (type_layer[i][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			double &neuron = this->neuron[0][i][h][j];

			if (strstr(type_layer[i], "ce")){
				if (strstr(type_layer[i], "sm")){
					// neuron = neuron;
				}
				else{
					neuron = 1 / (1 + exp(-neuron));
				}
			}
			else
			if (strstr(type_layer[i], "mse")){
				if (strstr(type_layer[i], "ht")){
					neuron = 2 / (1 + exp(-2 * neuron)) - 1;
				}
				else
				if (strstr(type_layer[i], "ia")){
					// neuron = neuron;
				}
				else{
					neuron = 1 / (1 + exp(-neuron));
				}
			}
		}
	}
}
void Multilayer_Perceptron::Adjust_Parameter(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	if (strstr(type_layer[i], "bn")){
		Batch_Normalization_Adjust_Parameter(i, j);
	}

	for (int h = 0; h < batch_size; h++){
		double *derivative = this->derivative[0][i][h];
		double *lower_neuron = this->neuron[0][i - 1][h];

		for (int k = 0; k < number_neurons[i - 1]; k++){
			weight[i][j][k] -= derivative[j] * lower_neuron[k];
		}
		weight[i][j][number_neurons[i - 1]] -= derivative[j];
	}
}
void Multilayer_Perceptron::Backpropagate(int layer_index, int neuron_index){
	if (layer_index == number_layers - 1){
		return;
	}

	int i = layer_index;
	int j = neuron_index;

	for (int h = 0; h < batch_size; h++){
		double sum = 0;

		double *derivative = this->derivative[0][i][h];
		double *upper_derivative = this->derivative[0][i + 1][h];

		for (int k = 0; k < number_neurons[i + 1]; k++){
			sum += upper_derivative[k] * weight[i + 1][k][j];
		}
		derivative[j] = sum;
	}
}
void Multilayer_Perceptron::Differentiate(int layer_index, int neuron_index, double learning_rate, double **target_output){
	int i = layer_index;
	int j = neuron_index;

	if (type_layer[i][0] == 'C'){
		for (int h = 0; h < batch_size; h++){
			double &derivative = this->derivative[0][i][h][j];
			double &neuron = this->neuron[0][i][h][j];

			if (strstr(type_layer[i], "ht")){
				derivative *= (1 - neuron) * (1 + neuron);
			}
			else
			if (strstr(type_layer[i], "ls")){
				derivative *= (1 - neuron) * neuron;
			}
			else{
				derivative *= (neuron > 0);
			}
		}

		if (strstr(type_layer[i], "bn")){
			Batch_Normalization_Differentiate(i, j);
		}
	}
	else
	if (type_layer[i][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			double &derivative = this->derivative[0][i][h][j];
			double &neuron = this->neuron[0][i][h][j];

			derivative = learning_rate * (neuron - target_output[h][j]);

			if (strstr(type_layer[i], "ce")){
				if (strstr(type_layer[i], "sm")){
					// derivative = derivative;
				}
				else{
					derivative *= (1 - neuron) * neuron;
				}
			}
			else
			if (strstr(type_layer[i], "mse")){
				if (strstr(type_layer[i], "ht")){
					derivative *= (1 - neuron) * (1 + neuron);
				}
				else
				if (strstr(type_layer[i], "ia")){
					// derivative *= 1;
				}
				else{
					derivative *= (1 - neuron) * neuron;
				}
			}
		}
	}
}
void Multilayer_Perceptron::Feedforward(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	for (int h = 0; h < batch_size; h++){
		double sum = 0;

		double *lower_neuron = this->neuron[0][i - 1][h];
		double *neuron = this->neuron[0][i][h];

		for (int k = 0; k < number_neurons[i - 1]; k++){
			sum += lower_neuron[k] * weight[i][j][k];
		}
		neuron[j] = sum + weight[i][j][number_neurons[i - 1]];
	}
}
void Multilayer_Perceptron::Softmax(int layer_index){
	int i = layer_index;

	if (strstr(type_layer[i], "sm")){
		for (int h = 0; h < batch_size; h++){
			double max = 0;
			double sum = 0;

			double *neuron = this->neuron[0][i][h];

			for (int j = 0; j < number_neurons[i]; j++){
				if (max < neuron[j]){
					max = neuron[j];
				}
			}
			for (int j = 0; j < number_neurons[i]; j++){
				neuron[j] = exp(neuron[j] - max);

				sum += neuron[j];
			}
			for (int j = 0; j < number_neurons[i]; j++){
				neuron[j] /= sum;
			}
		}
	}
}

void Multilayer_Perceptron::Batch_Normalization_Activate(char option[], int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double gamma = this->gamma[i][j];
	double beta = this->beta[i][j];
	double &mean = this->mean[i][j];
	double &variance = this->variance[i][j];
	double &sum_mean = this->sum_mean[i][j];
	double &sum_variance = this->sum_variance[i][j];

	double **neuron = this->neuron[0][i];
	double **neuron_batch[2] = { this->neuron[1][i], this->neuron[2][i] };

	if (!strcmp(option, "train")){
		double sum = 0;

		for (int h = 0; h < batch_size; h++){
			sum += neuron[h][j];
		}
		sum_mean += (mean = sum / batch_size);

		sum = 0;
		for (int h = 0; h < batch_size; h++){
			sum += (neuron[h][j] - mean) * (neuron[h][j] - mean);
		}
		sum_variance += (variance = sum / batch_size);

		for (int h = 0; h < batch_size; h++){
			neuron_batch[0][h][j] = (neuron[h][j] - mean) / sqrt(variance + epsilon);
			neuron_batch[1][h][j] = neuron[h][j];

			neuron[h][j] = gamma * neuron_batch[0][h][j] + beta;
		}
	}
	else
	if (!strcmp(option, "test")){
		double stdv = sqrt(variance + epsilon);

		for (int h = 0; h < batch_size; h++){
			neuron[h][j] = gamma / stdv * neuron[h][j] + (beta - gamma * mean / stdv);
		}
	}
}
void Multilayer_Perceptron::Batch_Normalization_Adjust_Parameter(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double sum = 0;

	double **derivative_batch = this->derivative[2][i];
	double **neuron_batch = this->neuron[1][i];

	for (int h = 0; h < batch_size; h++){
		sum += derivative_batch[h][j] * neuron_batch[h][j];
	}
	gamma[i][j] -= sum;

	sum = 0;
	for (int h = 0; h < batch_size; h++){
		sum += derivative_batch[h][j];
	}
	beta[i][j] -= sum;
}
void Multilayer_Perceptron::Batch_Normalization_Differentiate(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double derivative_mean;
	double derivative_variance;
	double sum = 0;

	double gamma = this->gamma[i][j];
	double beta = this->beta[i][j];
	double mean = this->mean[i][j];
	double variance = this->variance[i][j];

	double **derivative = this->derivative[0][i];
	double **derivative_batch[2] = { this->derivative[1][i], this->derivative[2][i] };
	double **neuron_batch[2] = { this->neuron[1][i], this->neuron[2][i] };

	for (int h = 0; h < batch_size; h++){
		derivative_batch[0][h][j] = derivative[h][j] * gamma;
		sum += derivative_batch[0][h][j] * (neuron_batch[1][h][j] - mean);
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, -1.5);

	sum = 0;
	for (int h = 0; h < batch_size; h++){
		sum += derivative_batch[0][h][j];
	}
	derivative_mean = -sum / sqrt(variance + epsilon);

	for (int h = 0; h < batch_size; h++){
		derivative_batch[1][h][j] = derivative[h][j];

		derivative[h][j] = derivative_batch[0][h][j] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_batch[1][h][j] - mean) / batch_size + derivative_mean / batch_size;
	}
}

void Multilayer_Perceptron::Resize_Memory(int batch_size){
	if (this->batch_size != batch_size){
		for (int g = 0; g < number_memory_types; g++){
			for (int i = 0; i < number_layers; i++){
				if (Access_Memory(g, i)){
					for (int h = 0; h < this->batch_size; h++){
						delete[] derivative[g][i][h];
						delete[] neuron[g][i][h];
					}
					derivative[g][i] = (double**)realloc(derivative[g][i], sizeof(double*)* batch_size);
					neuron[g][i] = (double**)realloc(neuron[g][i], sizeof(double*)* batch_size);

					for (int h = 0; h < batch_size; h++){
						derivative[g][i][h] = new double[number_neurons[i]];
						neuron[g][i][h] = new double[number_neurons[i]];
					}
				}
			}
		}
		this->batch_size = batch_size;
	}
}

bool Multilayer_Perceptron::Access_Memory(int type_index, int layer_index){
	int g = type_index;
	int i = layer_index;

	return (g == 0 || strstr(type_layer[i], "bn"));
}

Multilayer_Perceptron::Multilayer_Perceptron(string path) {
	ifstream file(path);

	if (file.is_open()) {
		file >> number_layers;
		batch_size			= 1;
		number_memory_types = 3;

		type_layer = new char*[number_layers];
		for (int i = 0; i < number_layers; i++) {
			string type;

			file >> type;
			strcpy(type_layer[i] = new char[type.size() + 1], type.c_str());
		}

		number_neurons = new int[number_layers];
		for (int i = 0; i < number_layers; i++) {
			file >> number_neurons[i];
		}
		file >> epsilon;

		gamma		 = new double*[number_layers];
		beta		 = new double*[number_layers];
		mean		 = new double*[number_layers];
		variance	 = new double*[number_layers];
		sum_mean	 = new double*[number_layers];
		sum_variance = new double*[number_layers];
		weight		 = new double**[number_layers];

		for (int i = 1; i < number_layers; i++) {
			if (strstr(this->type_layer[i], "bn")) {
				gamma[i]		= new double[number_neurons[i]];
				beta[i]			= new double[number_neurons[i]];
				mean[i]			= new double[number_neurons[i]];
				variance[i]		= new double[number_neurons[i]];
				sum_mean[i]		= new double[number_neurons[i]];
				sum_variance[i] = new double[number_neurons[i]];

				for (int j = 0; j < number_neurons[i]; j++) file >> gamma[i][j];
				for (int j = 0; j < number_neurons[i]; j++) file >> beta[i][j];
				for (int j = 0; j < number_neurons[i]; j++) file >> mean[i][j];
				for (int j = 0; j < number_neurons[i]; j++) file >> variance[i][j];
			}

			weight[i] = new double*[number_neurons[i]];

			for (int j = 0; j < number_neurons[i]; j++) {
				weight[i][j] = new double[number_neurons[i - 1] + 1];

				for (int k = 0; k < number_neurons[i - 1] + 1; k++) {
					file >> weight[i][j][k];
				}
			}
		}

		derivative	= new double***[number_memory_types];
		neuron		= new double***[number_memory_types];

		for (int g = 0; g < number_memory_types; g++) {
			derivative[g]	= new double**[number_layers];
			neuron[g]		= new double**[number_layers];

			for (int i = 0; i < number_layers; i++) {
				if (Access_Memory(g, i)) {
					derivative[g][i] = new double*[batch_size];
					neuron[g][i]	 = new double*[batch_size];

					for (int h = 0; h < batch_size; h++) {
						derivative[g][i][h] = new double[number_neurons[i]];
						neuron[g][i][h]		= new double[number_neurons[i]];
					}
				}
			}
		}
		file.close();
	}
	else {
		cerr << "[Multilayer_Perceptron], " + path + " not found" << endl;
	}
}
Multilayer_Perceptron::Multilayer_Perceptron(string type_layer[], int number_layers, int number_neurons[]){
	this->number_layers	= number_layers;
	this->number_neurons = new int[number_layers];
	this->type_layer	 = new char*[number_layers];

	batch_size			= 1;
	number_memory_types = 3;

	for (int i = 0; i < number_layers; i++){
		strcpy(this->type_layer[i] = new char[type_layer[i].size() + 1], type_layer[i].c_str());
		this->number_neurons[i] = number_neurons[i];
	}

	gamma		 = new double*[number_layers];
	beta		 = new double*[number_layers];
	mean		 = new double*[number_layers];
	variance	 = new double*[number_layers];
	sum_mean	 = new double*[number_layers];
	sum_variance = new double*[number_layers];
	weight		 = new double**[number_layers];

	for (int i = 1; i < number_layers; i++){
		if (strstr(this->type_layer[i], "bn")){
			gamma[i]		= new double[number_neurons[i]];
			beta[i]			= new double[number_neurons[i]];
			mean[i]			= new double[number_neurons[i]];
			variance[i]		= new double[number_neurons[i]];
			sum_mean[i]		= new double[number_neurons[i]];
			sum_variance[i] = new double[number_neurons[i]];
		}

		weight[i] = new double*[number_neurons[i]];

		for (int j = 0; j < number_neurons[i]; j++) {
			weight[i][j] = new double[number_neurons[i - 1] + 1];
		}
	}

	derivative	= new double***[number_memory_types];
	neuron		= new double***[number_memory_types];

	for (int g = 0; g < number_memory_types; g++){
		derivative[g]	= new double**[number_layers];
		neuron[g]		= new double**[number_layers];

		for (int i = 0; i < number_layers; i++){
			if (Access_Memory(g, i)){
				derivative[g][i] = new double*[batch_size];
				neuron[g][i]	 = new double*[batch_size];

				for (int h = 0; h < batch_size; h++){
					derivative[g][i][h] = new double[number_neurons[i]];
					neuron[g][i][h]		= new double[number_neurons[i]];
				}
			}
		}
	}
}
Multilayer_Perceptron::~Multilayer_Perceptron(){
	for (int i = 1; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			delete[] gamma[i];
			delete[] beta[i];
			delete[] mean[i];
			delete[] variance[i];
			delete[] sum_mean[i];
			delete[] sum_variance[i];
		}

		for (int j = 0; j < number_neurons[i]; j++) {
			delete[] weight[i][j];
		}
		delete[] weight[i];
	}
	delete[] gamma;
	delete[] beta;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;
	delete[] weight;

	for (int g = 0; g < number_memory_types; g++) {
		for (int i = 0; i < number_layers; i++) {
			if (Access_Memory(g, i)) {
				for (int h = 0; h < batch_size; h++) {
					delete[] derivative[g][i][h];
					delete[] neuron[g][i][h];
				}
				delete[] derivative[g][i];
				delete[] neuron[g][i];
			}
		}
		delete[] derivative[g];
		delete[] neuron[g];
	}
	delete[] derivative;
	delete[] neuron;

	for (int i = 0; i < number_layers; i++) {
		delete[] type_layer[i];
	}
	delete[] type_layer;
	delete[] number_neurons;
}

void Multilayer_Perceptron::Initialize_Parameter(double scale, double shift, int seed){
	srand(seed);

	for (int i = 1; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")){
			for (int j = 0; j < number_neurons[i]; j++){
				gamma[i][j] = 1;
				beta[i][j]	= 0;
			}
		}
		for (int j = 0; j < number_neurons[i]; j++){
			for (int k = 0; k < number_neurons[i - 1] + 1; k++){
				weight[i][j][k] = scale * rand() / RAND_MAX + shift;
			}
		}
	}
}
void Multilayer_Perceptron::Save_Model(string path){
	ofstream file(path);

	file << number_layers << endl;

	for (int i = 0; i < number_layers; i++) {
		file << type_layer[i] << endl;
	}
	for (int i = 0; i < number_layers; i++) {
		file << number_neurons[i] << endl;
	}
	file << epsilon << endl;

	for (int i = 1; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int j = 0; j < number_neurons[i]; j++) file << gamma[i][j] << endl;
			for (int j = 0; j < number_neurons[i]; j++) file << beta[i][j] << endl;
			for (int j = 0; j < number_neurons[i]; j++) file << mean[i][j] << endl;
			for (int j = 0; j < number_neurons[i]; j++) file << variance[i][j] << endl;
		}

		for (int j = 0; j < number_neurons[i]; j++) {
			for (int k = 0; k < number_neurons[i - 1] + 1; k++) {
				file << weight[i][j][k] << endl;
			}
		}
	}
	file.close();
}
void Multilayer_Perceptron::Test(double input[], double output[]){
	Resize_Memory(1);

	memcpy(neuron[0][0][0], input, sizeof(double) * number_neurons[0]);

	for (int i = 1; i < number_layers; i++){
		#pragma omp parallel for
		for (int j = 0; j < number_neurons[i]; j++){
			Feedforward(i, j);
			Activate("test", i, j);
		}
		Softmax(i);
	}
	memcpy(output, neuron[0][number_layers - 1][0], sizeof(double) * number_neurons[number_layers - 1]);
}
void Multilayer_Perceptron::Test(int batch_size, double **input, double **output) {
	Resize_Memory(batch_size);

	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		memcpy(neuron[0][0][h], input[h], sizeof(double) * number_neurons[0]);
	}
	for (int i = 1; i < number_layers; i++) {
		#pragma omp parallel for
		for (int j = 0; j < number_neurons[i]; j++) {
			Feedforward(i, j);
			Activate("test", i, j);
		}
		Softmax(i);
	}
	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		memcpy(output[h], neuron[0][number_layers - 1][h], sizeof(double) * number_neurons[number_layers - 1]);
	}
}

double Multilayer_Perceptron::Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output){
	int *index = new int[number_training];

	double loss = 0;

	double **target_output_batch = new double*[batch_size];

	for (int i = 0; i < number_training; i++){
		index[i] = i;
	}
	for (int i = 0; i < number_training; i++){
		int j = rand() % number_training;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;
	}
	Resize_Memory(batch_size);

	for (int i = 0; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")){
			memset(sum_mean[i],		0, sizeof(double) * number_neurons[i]);
			memset(sum_variance[i], 0, sizeof(double) * number_neurons[i]);
		}
	}
	this->epsilon = epsilon;

	for (int g = 0, h = 0; g < number_training; g++){
		memcpy(neuron[0][0][h], input[index[g]], sizeof(double) * number_neurons[0]);
		target_output_batch[h] = target_output[index[g]];

		if (++h == batch_size){
			h = 0;

			for (int i = 1; i < number_layers; i++){
				#pragma omp parallel for
				for (int j = 0; j < number_neurons[i]; j++){
					Feedforward(i, j);
					Activate("train", i, j);
				}
				Softmax(i);
			}
			for (int i = number_layers - 1; i > 0; i--){
				#pragma omp parallel for
				for (int j = 0; j < number_neurons[i]; j++){
					Backpropagate(i, j);
					Differentiate(i, j, learning_rate, target_output_batch);
				}
			}
			for (int i = number_layers - 1; i > 0; i--){
				#pragma omp parallel for
				for (int j = 0; j < number_neurons[i]; j++){
					Adjust_Parameter(i, j);
				}
			}

			for (int h = 0; h < batch_size; h++){
				for (int i = number_layers - 1, j = 0; j < number_neurons[i]; j++){
					if (strstr(type_layer[i], "ce")){
						loss -= target_output_batch[h][j] * log(neuron[0][i][h][j] + 0.000001) + (1 - target_output_batch[h][j]) * log(1 - neuron[0][i][h][j] + 0.000001);
					}
					if (strstr(type_layer[i], "mse")){
						loss += 0.5 * (neuron[0][i][h][j] - target_output_batch[h][j]) * (neuron[0][i][h][j] - target_output_batch[h][j]);
					}
				}
			}
		}
	}

	for (int i = 0; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")){
			for (int j = 0; j < number_neurons[i]; j++){
				mean[i][j]		= sum_mean[i][j] / (number_training / batch_size);
				variance[i][j]	= ((double)batch_size / (batch_size - 1)) * sum_variance[i][j] / (number_training / batch_size);
			}
		}
	}

	delete[] index;
	delete[] target_output_batch;

	return loss / number_training;
}
double Multilayer_Perceptron::Train(int batch_size, int number_training, int length_data[], double epsilon, double learning_rate, double ***input, double ***target_output) {
	int number_batches = 0;

	int ***index = new int**[number_training];

	double loss = 0;

	double **target_output_batch = new double*[batch_size];

	for (int i = 0; i < number_training; i++) {
		index[i] = new int*[length_data[i]];

		for (int j = 0; j < length_data[i]; j++) {
			index[i][j] = new int[2];
			index[i][j][0] = i;
			index[i][j][1] = j;
		}
	}
	for (int i = 0; i < number_training; i++) {
		for (int j = 0; j < length_data[i]; j++) {
			int k = rand() % number_training;
			int l = rand() % length_data[k];

			int temp[2] = { index[i][j][0], index[i][j][1] };

			index[i][j][0] = index[k][l][0];
			index[i][j][1] = index[k][l][1];
			index[k][l][0] = temp[0];
			index[k][l][1] = temp[1];
		}
	}

	Resize_Memory(batch_size);
	this->epsilon = epsilon;

	for (int i = 0; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			memset(sum_mean[i],		0, sizeof(double) * number_neurons[i]);
			memset(sum_variance[i], 0, sizeof(double) * number_neurons[i]);
		}
	}

	for (int g = 0, h = 0; g < number_training; g++) {
		for (int s = 0; s < length_data[g]; s++) {
			memcpy(neuron[0][0][h], input[index[g][s][0]][index[g][s][1]], sizeof(double) * number_neurons[0]);
			target_output_batch[h] = target_output[index[g][s][0]][index[g][s][1]];

			if (++h == batch_size) {
				number_batches++;
				h = 0;

				for (int i = 1; i < number_layers; i++) {
					#pragma omp parallel for
					for (int j = 0; j < number_neurons[i]; j++) {
						Feedforward(i, j);
						Activate("train", i, j);
					}
					Softmax(i);
				}
				for (int i = number_layers - 1; i > 0; i--) {
					#pragma omp parallel for
					for (int j = 0; j < number_neurons[i]; j++) {
						Backpropagate(i, j);
						Differentiate(i, j, learning_rate, target_output_batch);
					}
				}
				for (int i = number_layers - 1; i > 0; i--) {
					#pragma omp parallel for
					for (int j = 0; j < number_neurons[i]; j++) {
						Adjust_Parameter(i, j);
					}
				}

				for (int h = 0; h < batch_size; h++) {
					for (int i = number_layers - 1, j = 0; j < number_neurons[i]; j++) {
						if (strstr(type_layer[i], "ce")) {
							loss -= target_output_batch[h][j] * log(neuron[0][i][h][j] + 0.000001) + (1 - target_output_batch[h][j]) * log(1 - neuron[0][i][h][j] + 0.000001);
						}
						if (strstr(type_layer[i], "mse")) {
							loss += 0.5 * (neuron[0][i][h][j] - target_output_batch[h][j]) * (neuron[0][i][h][j] - target_output_batch[h][j]);
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int j = 0; j < number_neurons[i]; j++) {
				mean[i][j]		= sum_mean[i][j] / number_batches;
				variance[i][j]	= ((double)batch_size / (batch_size - 1)) * sum_variance[i][j] / number_batches;
			}
		}
	}

	for (int i = 0; i < number_training; i++) {
		for (int t = 0; t < length_data[i]; t++) {
			delete[] index[i][t];
		}
		delete[] index[i];
	}
	delete[] index;
	delete[] target_output_batch;

	return loss / (batch_size * number_batches);
}