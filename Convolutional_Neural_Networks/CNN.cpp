#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "CNN.h"

void Convolutional_Neural_Networks::Activate(char option[], int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	map_index *= map_area[i];

	if (type_layer[i][0] == 'C'){
		if (strstr(type_layer[i], "bn")) {
			Batch_Normalization_Activate(option, i, j);
		}

		for (int h = 0; h < batch_size; h++){
			double mask = 1;

			if (strstr(type_layer[i], "do")){
				char *rate = strstr(type_layer[i], "do") + 2;

				if (!strcmp(option, "train")){
					mask = ((double)rand() / RAND_MAX <= atof(rate));
				}
				else
				if (!strcmp(option, "test")){
					mask = atof(rate);
				}
			}

			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double &neuron = this->neuron[0][i][h][k];

				if (strstr(type_layer[i], "ht")) {
					neuron = 2 / (1 + exp(-2 * neuron)) - 1;
				}
				else
				if (strstr(type_layer[i], "ls")) {
					neuron = 1 / (1 + exp(-neuron));
				}
				else {
					neuron *= (neuron > 0);
				}

				// dropout
				neuron *= mask;
			}
		}
	}
	else
	if (type_layer[i][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double &neuron = this->neuron[0][i][h][k];

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
}
void Convolutional_Neural_Networks::Adjust_Parameter(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		if (strstr(type_layer[i], "bn")){
			Batch_Normalization_Adjust_Parameter(layer_index, map_index);
		}
		map_index *= map_area[i];

		for (int h = 0; h < batch_size; h++){
			double *derivative = this->derivative[0][i][h];
			double *lower_neuron = this->neuron[0][i - 1][h];

			for (int k = map_index, index = 0; k < map_index + map_area[i]; k++) {
				vector<node> *connected_neuron = &(this->connected_neuron[i][k]);

				for (auto connection = (*connected_neuron).begin(); connection != (*connected_neuron).end(); connection++) {
					(*connection->weight) -= derivative[k] * lower_neuron[connection->index];
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Backpropagate(int layer_index, int map_index){
	if (layer_index == number_layers - 1){
		return;
	}

	int i = layer_index;
	int j = map_index;

	map_index *= map_area[i];

	if (type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			double *upper_derivative = this->derivative[0][i + 1][h];

			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double sum = 0;

				vector<node> *connected_derivative = &(this->connected_derivative[i][k]);

				for (auto connection = (*connected_derivative).begin(); connection != (*connected_derivative).end(); connection++) {
					sum += upper_derivative[connection->index] * (*connection->weight);
				}
				derivative[0][i][h][k] = sum;
			}
		}
	}
	else
	if (type_layer[i + 1][0] == 'P'){
		for (int h = 0; h < batch_size; h++){
			double *upper_derivative = this->derivative[0][i + 1][h];

			for (int k = map_index; k < map_index + map_area[i]; k++) {
				derivative[0][i][h][k] = upper_derivative[connected_derivative[i][k].begin()->index];
			}
		}
	}
}
void Convolutional_Neural_Networks::Differentiate(int layer_index, int map_index, double learning_rate, double **target_output){
	int i = layer_index;
	int j = map_index;

	map_index *= map_area[i];

	if (type_layer[i][0] == 'C'){
		for (int h = 0; h < batch_size; h++){
			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double &derivative = this->derivative[0][i][h][k];
				double &neuron = this->neuron[0][i][h][k];

				if (strstr(type_layer[i], "ht")) {
					derivative *= (1 - neuron) * (1 + neuron);
				}
				else
				if (strstr(type_layer[i], "ls")) {
					derivative *= (1 - neuron) * neuron;
				}
				else {
					derivative *= (neuron > 0);
				}
			}
		}

		if (strstr(type_layer[i], "bn")) {
			Batch_Normalization_Differentiate(i, j);
		}
	}
	else
	if (type_layer[i][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double &derivative = this->derivative[0][i][h][k];
				double &neuron = this->neuron[0][i][h][k];

				derivative = learning_rate * (neuron - target_output[h][k]);

				if (strstr(type_layer[i], "ce")) {
					if (strstr(type_layer[i], "sm")) {
						// derivative = derivative;
					}
					else {
						// derivative = derivative;
					}
				}
				else
				if (strstr(type_layer[i], "mse")) {
					if (strstr(type_layer[i], "ht")) {
						derivative *= (1 - neuron) * (1 + neuron);
					}
					else
					if (strstr(type_layer[i], "ia")) {
						// derivative *= 1;
					}
					else {
						derivative *= (1 - neuron) * neuron;
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Feedforward(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	map_index *= map_area[i];

	if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		for (int h = 0; h < batch_size; h++){
			double *lower_neuron = this->neuron[0][i - 1][h];

			for (int k = map_index; k < map_index + map_area[i]; k++) {
				double sum = 0;

				vector<node> *connected_neuron = &(this->connected_neuron[i][k]);

				for (auto connection = (*connected_neuron).begin(); connection != (*connected_neuron).end(); connection++) {
					sum += lower_neuron[connection->index] * (*connection->weight);
				}
				neuron[0][i][h][k] = sum;
			}
		}
	}
	else
	if (type_layer[i][0] == 'P'){
		for (int h = 0; h < batch_size; h++){
			double *lower_neuron = this->neuron[0][i - 1][h];

			for (int k = map_index; k < map_index + map_area[i]; k++) {
				vector<node> *connected_neuron = &(this->connected_neuron[i][k]);

				if (strstr(type_layer[i], "avg")) {
					double sum = 0;

					for (auto connection = (*connected_neuron).begin(); connection != (*connected_neuron).end(); connection++) {
						sum += lower_neuron[connection->index];
					}
					neuron[0][i][h][k] = sum / (*connected_neuron).size();
				}
				else
				if (strstr(type_layer[i], "max")) {
					double max = -1;

					for (auto connection = (*connected_neuron).begin(); connection != (*connected_neuron).end(); connection++) {
						if (max < lower_neuron[connection->index]) {
							max = lower_neuron[connection->index];
						}
					}
					neuron[0][i][h][k] = max;
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Softmax(int layer_index){
	int i = layer_index;

	if (strstr(type_layer[i], "sm")){
		for (int h = 0; h < batch_size; h++){
			double max = 0;
			double sum = 0;

			double *neuron = this->neuron[0][i][h];

			for (int j = 0; j < number_nodes[i]; j++){
				if (max < neuron[j]){
					max = neuron[j];
				}
			}
			for (int j = 0; j < number_nodes[i]; j++){
				neuron[j] = exp(neuron[j] - max);
				sum += neuron[j];
			}
			for (int j = 0; j < number_nodes[i]; j++){
				neuron[j] /= sum;
			}
		}
	}
}

void Convolutional_Neural_Networks::Batch_Normalization_Activate(char option[], int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	double gamma = this->gamma[i][j];
	double beta = this->beta[i][j];
	double &mean = this->mean[i][j];
	double &variance = this->variance[i][j];
	double &sum_mean = this->sum_mean[i][j];
	double &sum_variance = this->sum_variance[i][j];

	double **neuron = this->neuron[0][i];
	double **neuron_batch[2] = { this->neuron[1][i], this->neuron[2][i] };

	j *= map_area[i];

	if (!strcmp(option, "train")){
		double sum = 0;

		for (int h = 0; h < batch_size; h++){
			for (int k = j; k < j + map_area[i]; k++){
				sum += neuron[h][k];
			}
		}
		sum_mean += (mean = sum / (batch_size * map_area[i]));

		sum = 0;
		for (int h = 0; h < batch_size; h++){
			for (int k = j; k < j + map_area[i]; k++){
				sum += (neuron[h][k] - mean) * (neuron[h][k] - mean);
			}
		}
		sum_variance += (variance = sum / (batch_size * map_area[i]));

		for (int h = 0; h < batch_size; h++){
			for (int k = j; k < j + map_area[i]; k++) {
				neuron_batch[0][h][k] = (neuron[h][k] - mean) / sqrt(variance + epsilon);
				neuron_batch[1][h][k] = neuron[h][k];

				neuron[h][k] = gamma * neuron_batch[0][h][k] + beta;
			}
		}
	}
	else
	if (!strcmp(option, "test")){
		double stdv = sqrt(variance + epsilon);

		for (int h = 0; h < batch_size; h++){
			for (int k = j; k < j + map_area[i]; k++){
				neuron[h][k] = gamma / stdv * neuron[h][k] + (beta - gamma * mean / stdv);
			}
		}
	}
}
void Convolutional_Neural_Networks::Batch_Normalization_Adjust_Parameter(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	double sum = 0;

	double &gamma = this->gamma[i][j];
	double &beta = this->beta[i][j];
	double **derivative_batch = this->derivative[2][i];
	double **neuron_batch = this->neuron[1][i];

	j *= map_area[i];

	for (int h = 0; h < batch_size; h++){
		for (int k = j; k < j + map_area[i]; k++){
			sum += derivative_batch[h][k] * neuron_batch[h][k];
		}
	}
	gamma -= sum;

	sum = 0;
	for (int h = 0; h < batch_size; h++){
		for (int k = j; k < j + map_area[i]; k++){
			sum += derivative_batch[h][k];
		}
	}
	beta -= sum;
}
void Convolutional_Neural_Networks::Batch_Normalization_Differentiate(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

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

	j *= map_area[i];

	for (int h = 0; h < batch_size; h++){
		for (int k = j; k < j + map_area[i]; k++){
			derivative_batch[0][h][k] = derivative[h][k] * gamma;
			sum += derivative_batch[0][h][k] * (neuron_batch[1][h][k] - mean);
		}
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, -1.5);

	sum = 0;
	for (int h = 0; h < batch_size; h++){
		for (int k = j; k < j + map_area[i]; k++){
			sum += derivative_batch[0][h][k];
		}
	}
	derivative_mean = -sum / sqrt(variance + epsilon);

	for (int h = 0; h < batch_size; h++){
		for (int k = j; k < j + map_area[i]; k++){
			derivative_batch[1][h][k] = derivative[h][k];

			derivative[h][k] = derivative_batch[0][h][k] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_batch[1][h][k] - mean) / (batch_size * map_area[i]) + derivative_mean / (batch_size * map_area[i]);
		}
	}
}

void Convolutional_Neural_Networks::Construct_Networks() {
	batch_size			= 1;
	number_memory_types = 3;

	kernel_width	= new int[number_layers];
	kernel_height	= new int[number_layers];
	map_area		= new int[number_layers];
	number_nodes	= new int[number_layers];
	stride_width	= new int[number_layers];
	stride_height	= new int[number_layers];

	for (int i = 0; i < number_layers; i++) {
		map_area[i]		= map_height[i] * map_width[i];
		number_nodes[i] = number_maps[i] * map_area[i];

		if (strstr(type_layer[i], "ks")) {
			char *kernel_size = strstr(type_layer[i], "ks");

			kernel_width[i]	 = atoi(kernel_size + 2);
			kernel_size		 = strstr(kernel_size, ",");
			kernel_height[i] = (kernel_size && atoi(kernel_size + 1) > 0) ? (atoi(kernel_size + 1)) : (kernel_width[i]);
		}
		else {
			kernel_width[i]  = (i == 0 || type_layer[i][0] == 'P') ? (0) : (abs(map_width[i - 1] - map_width[i]) + 1);
			kernel_height[i] = (i == 0 || type_layer[i][0] == 'P') ? (0) : (abs(map_height[i - 1] - map_height[i]) + 1);
		}

		if (strstr(this->type_layer[i], "st")) {
			char *stride = strstr(type_layer[i], "st");

			stride_width[i]	 = atoi(stride + 2);
			stride			 = strstr(stride, ",");
			stride_height[i] = (stride && atoi(stride + 1) > 0) ? (atoi(stride + 1)) : (stride_width[i]);
		}
		else {
			stride_width[i]  = (type_layer[i][0] == 'P') ? ((map_width[i - 1] > map_width[i]) ? (map_width[i - 1] / map_width[i]) : (map_width[i] / map_width[i - 1])) : (1);
			stride_height[i] = (type_layer[i][0] == 'P') ? ((map_height[i - 1] > map_height[i]) ? (map_height[i - 1] / map_height[i]) : (map_height[i] / map_height[i - 1])) : (1);
		}
	}

	gamma		 = new double*[number_layers];
	beta		 = new double*[number_layers];
	mean		 = new double*[number_layers];
	variance	 = new double*[number_layers];
	sum_mean	 = new double*[number_layers];
	sum_variance = new double*[number_layers];
	weight		 = new double***[number_layers];

	for (int i = 1; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")){
			gamma[i]		= new double[number_maps[i]];
			beta[i]			= new double[number_maps[i]];
			mean[i]			= new double[number_maps[i]];
			variance[i]		= new double[number_maps[i]];
			sum_mean[i]		= new double[number_maps[i]];
			sum_variance[i] = new double[number_maps[i]];
		}
		if (kernel_width[i]) {
			bool depthwise_separable = (strstr(type_layer[i], "dw") != 0);

			weight[i] = new double**[number_maps[i]];

			for (int j = 0; j < number_maps[i]; j++) {
				weight[i][j] = new double*[number_maps[i - 1] + 1];

				for (int k = 0; k < number_maps[i - 1] + 1; k++) {
					weight[i][j][k] = (!depthwise_separable || j == k || k == number_maps[i - 1]) ? (new double[kernel_height[i] * kernel_width[i]]) : (nullptr);
				}
			}
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
					derivative[g][i][h] = new double[number_nodes[i] + 1];
					neuron[g][i][h]		= new double[number_nodes[i] + 1];
					neuron[g][i][h][number_nodes[i]] = 1;
				}
			}
		}
	}

	connected_derivative = new vector<node>*[number_layers];
	connected_neuron	 = new vector<node>*[number_layers];
	
	for (int i = 0; i < number_layers; i++) {
		connected_derivative[i] = new vector<node>[number_nodes[i]];
		connected_neuron[i]		= new vector<node>[number_nodes[i]];
	}

	for (int i = 1; i < number_layers; i++) {
		for (int j = 0; j < number_maps[i]; j++) {
			for (int k = 0; k < map_height[i]; k++) {
				for (int l = 0; l < map_width[i]; l++) {
					int index[2] = { j * map_area[i] + k * map_width[i] + l, 0 };

					node connection;

					if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L') {
						for (int m = 0; m < number_maps[i - 1]; m++) {
							if (weight[i][j][m]) {
								for (int n = 0; n < map_height[i - 1]; n++) {
									for (int o = 0; o < map_width[i - 1]; o++) {
										int distance[2] = { (map_height[i] < map_height[i - 1]) ? (n - k * stride_height[i]) : (k - n * stride_height[i]) , (map_width[i] < map_width[i - 1]) ? (o - l * stride_width[i]) : (l - o * stride_width[i]) };

										if (0 <= distance[0] && distance[0] < kernel_height[i] && 0 <= distance[1] && distance[1] < kernel_width[i]) {
											index[1] = m * map_area[i - 1] + n * map_width[i - 1] + o;

											connection.index  = index[0];
											connection.weight = &weight[i][j][m][distance[0] * kernel_width[i] + distance[1]];
											connected_derivative[i - 1][index[1]].push_back(connection);

											connection.index  = index[1];
											connection.weight = &weight[i][j][m][distance[0] * kernel_width[i] + distance[1]];
											connected_neuron[i][index[0]].push_back(connection);
										}
									}
								}
							}
						}
						connection.index  = number_nodes[i - 1];
						connection.weight = &weight[i][j][number_maps[i - 1]][0];
						connected_neuron[i][index[0]].push_back(connection);
					}
					else
					if (type_layer[i][0] == 'P') {
						for (int n = 0; n < map_height[i - 1]; n++) {
							for (int o = 0; o < map_width[i - 1]; o++) {
								int distance[2] = { (map_height[i] < map_height[i - 1]) ? (n - k * stride_height[i]) : (k - n * stride_height[i]) , (map_width[i] < map_width[i - 1]) ? (o - l * stride_width[i]) : (l - o * stride_width[i]) };

								if (0 <= distance[0] && distance[0] < ((kernel_height[i]) ? (kernel_height[i]) : (stride_height[i])) && 0 <= distance[1] && distance[1] < ((kernel_width[i]) ? (kernel_width[i]) : (stride_width[i]))) {
									index[1] = j * map_height[i - 1] * map_width[i - 1] + n * map_width[i - 1] + o;

									connection.index = index[0];
									connected_derivative[i - 1][index[1]].push_back(connection);

									connection.index = index[1];
									connected_neuron[i][index[0]].push_back(connection);
								}
							}
						}
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Resize_Memory(int batch_size){
	if (this->batch_size != batch_size){
		for (int g = 0; g < number_memory_types; g++){
			for (int i = 0; i < number_layers; i++){
				if (Access_Memory(g, i)){
					for (int h = 0; h < this->batch_size; h++){
						delete[] derivative[g][i][h];
						delete[] neuron[g][i][h];
					}
					derivative[g][i] = (double**)realloc(derivative[g][i], sizeof(double*)* batch_size);
					neuron[g][i]	 = (double**)realloc(neuron[g][i], sizeof(double*)* batch_size);

					for (int h = 0; h < batch_size; h++){
						derivative[g][i][h] = new double[number_nodes[i] + 1];
						neuron[g][i][h]		= new double[number_nodes[i] + 1];
						neuron[g][i][h][number_nodes[i]] = 1;
					}
				}
			}
		}
		this->batch_size = batch_size;
	}
}

bool Convolutional_Neural_Networks::Access_Memory(int type_index, int layer_index){
	int g = type_index;
	int i = layer_index;

	return (g == 0 || strstr(type_layer[i], "bn"));
}

Convolutional_Neural_Networks::Convolutional_Neural_Networks(string path) {
	ifstream file(path);

	if (file.is_open()) {
		file >> number_layers;

		type_layer = new char*[number_layers];
		for (int i = 0; i < number_layers; i++) {
			string type;

			file >> type;
			strcpy(type_layer[i] = new char[type.size() + 1], type.c_str());
		}

		number_maps = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> number_maps[i];
		map_width = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> map_width[i];
		map_height = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> map_height[i];
		file >> epsilon;

		Construct_Networks();
	}
}
Convolutional_Neural_Networks::Convolutional_Neural_Networks(string type_layer[], int number_layers, int number_maps[], int map_width[], int map_height[]){
	this->map_width		= new int[number_layers];
	this->map_height	= new int[number_layers];
	this->number_layers	= number_layers;
	this->number_maps	= new int[number_layers];
	this->type_layer	= new char*[number_layers];

	for (int i = 0; i < number_layers; i++){
		strcpy(this->type_layer[i] = new char[type_layer[i].size() + 1], type_layer[i].c_str());
		this->number_maps[i] = number_maps[i];
		this->map_width[i]	 = (map_width == nullptr) ? (1) : (map_width[i]);
		this->map_height[i]	 = (map_height == nullptr) ? (1) : (map_height[i]);
	}
	Construct_Networks();
}
Convolutional_Neural_Networks::~Convolutional_Neural_Networks(){
	for (int i = 1; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")) {
			delete[] gamma[i];
			delete[] beta[i];
			delete[] mean[i];
			delete[] variance[i];
			delete[] sum_mean[i];
			delete[] sum_variance[i];
		}
		if (kernel_width[i]) {
			bool depthwise_separable = (strstr(type_layer[i], "dw") != 0);

			for (int j = 0; j < number_maps[i]; j++) {
				for (int k = 0; k < number_maps[i - 1] + 1; k++) {
					if (weight[i][j][k]) {
						delete[] weight[i][j][k];
					}
				}
				delete[] weight[i][j];
			}
			delete[] weight[i];
		}
	}
	delete[] gamma;
	delete[] beta;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;
	delete[] weight;

	for (int g = 0; g < number_memory_types; g++){
		for (int i = 0; i < number_layers; i++){
			if (Access_Memory(g, i)){
				for (int h = 0; h < batch_size; h++){
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

	for (int i = 0; i < number_layers; i++){
		delete[] connected_derivative[i];
		delete[] connected_neuron[i];
		delete[] type_layer[i];
	}
	delete[] connected_derivative;
	delete[] connected_neuron;
	delete[] type_layer;

	delete[] kernel_width;
	delete[] kernel_height;
	delete[] map_width;
	delete[] map_height;
	delete[] number_maps;
	delete[] number_nodes;
	delete[] stride_width;
	delete[] stride_height;
}

void Convolutional_Neural_Networks::Initialize_Parameter(double scale, double shift, int seed){
	srand(seed);

	for (int i = 1; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int j = 0; j < number_maps[i]; j++) {
				gamma[i][j] = 1;
				beta[i][j]	= 0;
			}
		}
		if (kernel_width[i]) {
			for (int j = 0; j < number_maps[i]; j++) {
				for (int k = 0; k < number_maps[i - 1] + 1; k++) {
					if (weight[i][j][k]) {
						for (int l = 0; l < kernel_height[i] * kernel_width[i]; l++) {
							weight[i][j][k][l] = scale * rand() / RAND_MAX + shift;
						}
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Save_Model(string path){
	ofstream file(path);

	file << number_layers << endl;
	for (int i = 0; i < number_layers; i++) file << type_layer[i] << endl;
	for (int i = 0; i < number_layers; i++) file << number_maps[i] << endl;
	for (int i = 0; i < number_layers; i++) file << map_width[i] << endl;
	for (int i = 0; i < number_layers; i++) file << map_height[i] << endl;
	file << epsilon << endl;

	for (int i = 1; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int j = 0; j < number_maps[i]; j++) file << gamma[i][j] << endl;
			for (int j = 0; j < number_maps[i]; j++) file << beta[i][j] << endl;
			for (int j = 0; j < number_maps[i]; j++) file << mean[i][j] << endl;
			for (int j = 0; j < number_maps[i]; j++) file << variance[i][j] << endl;
		}
		if (kernel_width[i]) {
			for (int j = 0; j < number_maps[i]; j++) {
				for (int k = 0; k < number_maps[i - 1] + 1; k++) {
					if (weight[i][j][k]) {
						for (int l = 0; l < kernel_height[i] * kernel_width[i]; l++) {
							file << weight[i][j][k][l] << endl;
						}
					}
				}
			}
		}
	}
	file.close();
}
void Convolutional_Neural_Networks::Test(double input[], double output[]){
	Resize_Memory(1);

	memcpy(neuron[0][0][0], input, sizeof(double) * number_nodes[0]);

	for (int i = 1; i < number_layers; i++){
		#pragma omp parallel for
		for (int j = 0; j < number_maps[i]; j++) {
			Feedforward (i, j);
			Activate	("test", i, j);
		}
		Softmax(i);
	}
	memcpy(output, neuron[0][number_layers - 1][0], sizeof(double) * number_nodes[number_layers - 1]);
}

double Convolutional_Neural_Networks::Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output){
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
			memset(sum_mean[i], 0, sizeof(double) * number_maps[i]);
			memset(sum_variance[i], 0, sizeof(double) * number_maps[i]);
		}
	}
	this->epsilon = epsilon;

	for (int g = 0, h = 0; g < number_training; g++){
		memcpy(neuron[0][0][h], input[index[g]], sizeof(double) * number_nodes[0]);
		target_output_batch[h] = target_output[index[g]];

		if (++h == batch_size){
			h = 0;

			for (int i = 1; i < number_layers; i++){
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++) {
					Feedforward (i, j);
					Activate	("train", i, j);
				}
				Softmax(i);
			}

			for (int i = number_layers - 1; i > 0; i--){
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++) {
					Backpropagate(i, j);
					Differentiate(i, j, learning_rate, target_output_batch);
				}
			}
			for (int i = number_layers - 1; i > 0; i--){
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++){
					Adjust_Parameter(i, j);
				}
			}

			for (int h = 0, i = number_layers - 1; h < batch_size; h++){
				if (strstr(type_layer[i], "ce")) {
					for (int j = 0; j < number_nodes[i]; j++) {
						loss -= target_output_batch[h][j] * log(neuron[0][i][h][j] + 0.000001) + (1 - target_output_batch[h][j]) * log(1 - neuron[0][i][h][j] + 0.000001);
					}
				}
				if (strstr(type_layer[i], "mse")) {
					for (int j = 0; j < number_nodes[i]; j++) {
						loss += 0.5 * (neuron[0][i][h][j] - target_output_batch[h][j]) * (neuron[0][i][h][j] - target_output_batch[h][j]);
					}
				}
			}
		}
	}

	for (int i = 0; i < number_layers; i++){
		if (strstr(type_layer[i], "bn")){
			for (int j = 0; j < number_maps[i]; j++){
				mean[i][j] = sum_mean[i][j] / (number_training / batch_size);
				variance[i][j] = ((double)batch_size / (batch_size - 1)) * sum_variance[i][j] / (number_training / batch_size);
			}
		}
	}

	delete[] target_output_batch;
	delete[] index;

	return loss / number_training;
}
