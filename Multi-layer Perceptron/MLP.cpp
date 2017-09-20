#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "MLP.h"

void Multilayer_Perceptron::Activate(char option[], int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Activate(option, i, j);
		}

		for(int h = 0;h < batch_size;h++){
			double &neuron = this->neuron[0][i][h][j];

			if(strstr(type_layer[i], "ht")){
				neuron = 2 / (1 + exp(-2 * neuron)) - 1;
			}
			else
			if(strstr(type_layer[i], "ls")){
				neuron = 1 / (1 + exp(-neuron));
			}
			else{
				neuron *= (neuron > 0);
			}

			// Dropout *************************************************
			if(strstr(type_layer[i], "do")){
				char *rate = strstr(type_layer[i], "do") + 2;

				if(!strcmp(option, "train")){
					#pragma omp atomic
					neuron *= ((double)rand() / RAND_MAX <= atof(rate));
				}
				else
				if(!strcmp(option, "test")){
					neuron *= atof(rate);
				}
			}
			// *********************************************************
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			double &neuron = this->neuron[0][i][h][j];

			if(strstr(type_layer[i], "ce")){
				if(strstr(type_layer[i], "sm")){
					// neuron = neuron;
				}
				else{
					neuron = 1 / (1 + exp(-neuron));
				}
			}
			else
			if(strstr(type_layer[i], "mse")){
				if(strstr(type_layer[i], "ht")){
					neuron = 2 / (1 + exp(-2 * neuron)) - 1;
				}
				else
				if(strstr(type_layer[i], "ia")){
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

	if(strstr(type_layer[i], "bn")){
		Batch_Normalization_Adjust_Parameter(i, j);
	}
		
	for(int h = 0;h < batch_size;h++){
		double *derivative	 = this->derivative[0][i][h];
		double *lower_neuron = this->neuron[0][i - 1][h];

		for(int k = 0;k < number_neuron[i - 1];k++){
			weight[i][j][k] -= derivative[j] * lower_neuron[k];
		}
		weight[i][j][number_neuron[i - 1]] -= derivative[j];
	}
}
void Multilayer_Perceptron::Backpropagate(int layer_index, int neuron_index){
	if(layer_index == number_layer - 1){
		return;
	}

	int i = layer_index;
	int j = neuron_index;

	for(int h = 0;h < batch_size;h++){
		double sum = 0;

		double *derivative		 = this->derivative[0][i][h];
		double *upper_derivative = this->derivative[0][i + 1][h];

		for(int k = 0;k < number_neuron[i + 1];k++){
			sum += upper_derivative[k] * weight[i + 1][k][j];
		}
		derivative[j] = sum;
	}
}
void Multilayer_Perceptron::Differentiate(int layer_index, int neuron_index, double learning_rate, double **target_output){
	int i = layer_index;
	int j = neuron_index;

	if(type_layer[i][0] == 'C'){
		for(int h = 0;h < batch_size;h++){
			double &derivative	= this->derivative[0][i][h][j];
			double &neuron		= this->neuron[0][i][h][j];

			if(strstr(type_layer[i], "ht")){
				derivative *= (1 - neuron) * (1 + neuron);
			}
			else
			if(strstr(type_layer[i], "ls")){
				derivative *= (1 - neuron) * neuron;
			}
			else{
				derivative *= (neuron > 0);
			}
		}

		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Differentiate(i, j);
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			double &derivative	= this->derivative[0][i][h][j];
			double &neuron		= this->neuron[0][i][h][j];

			derivative = learning_rate * (neuron - target_output[h][j]);

			if(strstr(type_layer[i], "ce")){
				if(strstr(type_layer[i], "sm")){
					// derivative = derivative;
				}
				else{
					derivative *= (1 - neuron) * neuron;
				}
			}
			else
			if(strstr(type_layer[i], "mse")){
				if(strstr(type_layer[i], "ht")){
					derivative *= (1 - neuron) * (1 + neuron);
				}
				else
				if(strstr(type_layer[i], "ia")){
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

	for(int h = 0;h < batch_size;h++){
		double sum = 0;

		double *lower_neuron = this->neuron[0][i - 1][h];
		double *neuron		 = this->neuron[0][i][h];

		for(int k = 0;k < number_neuron[i - 1];k++){
			sum += lower_neuron[k] * weight[i][j][k];
		}
		neuron[j] = sum + weight[i][j][number_neuron[i - 1]];
	}
}
void Multilayer_Perceptron::Softmax(int layer_index){
	int i = layer_index;

	if(strstr(type_layer[i], "sm")){
		for(int h = 0;h < batch_size;h++){
			double max = 0;
			double sum = 0;

			double *neuron = this->neuron[0][i][h];

			for(int j = 0;j < number_neuron[i];j++){
				if(max < neuron[j]){
					max = neuron[j];
				}
			}
			for(int j = 0;j < number_neuron[i];j++){
				neuron[j] = exp(neuron[j] - max);

				sum += neuron[j];
			}
			for(int j = 0;j < number_neuron[i];j++){
				neuron[j] /= sum;
			}
		}
	}
}

void Multilayer_Perceptron::Batch_Normalization_Activate(char option[], int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double gamma		 = this->gamma[i][j];
	double beta			 = this->beta[i][j];
	double &mean		 = this->mean[i][j];
	double &variance	 = this->variance[i][j];
	double &sum_mean	 = this->sum_mean[i][j];
	double &sum_variance = this->sum_variance[i][j];

	double **neuron			 = this->neuron[0][i];
	double **neuron_batch[2] = {this->neuron[1][i], this->neuron[2][i]};

	if(!strcmp(option, "train")){
		double sum = 0;

		for(int h = 0;h < batch_size;h++){
			sum += neuron[h][j];
		}
		sum_mean += (mean = sum / batch_size);
							
		sum = 0;
		for(int h = 0;h < batch_size;h++){
			sum += (neuron[h][j] - mean) * (neuron[h][j] - mean);
		}
		sum_variance += (variance = sum / batch_size);
			
		for(int h = 0;h < batch_size;h++){
			neuron_batch[0][h][j] = (neuron[h][j] - mean) / sqrt(variance + epsilon);
			neuron_batch[1][h][j] = neuron[h][j];

			neuron[h][j] = gamma * neuron_batch[0][h][j] + beta;
		}
	}
	else
	if(!strcmp(option, "test")){
		double stdv = sqrt(variance + epsilon);

		for(int h = 0;h < batch_size;h++){						
			neuron[h][j] = gamma / stdv * neuron[h][j] + (beta - gamma * mean / stdv);
		}
	}
}
void Multilayer_Perceptron::Batch_Normalization_Adjust_Parameter(int layer_index, int neuron_index){
	int i = layer_index;
	int j = neuron_index;

	double sum = 0;

	double **derivative_batch	= this->derivative[2][i];
	double **neuron_batch		= this->neuron[1][i];
		
	for(int h = 0;h < batch_size;h++){
		sum += derivative_batch[h][j] * neuron_batch[h][j];
	}
	gamma[i][j] -= sum;
						
	sum = 0;
	for(int h = 0;h < batch_size;h++){
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

	double gamma	= this->gamma[i][j];
	double beta		= this->beta[i][j];
	double mean		= this->mean[i][j];
	double variance	= this->variance[i][j];

	double **derivative			 = this->derivative[0][i];
	double **derivative_batch[2] = {this->derivative[1][i], this->derivative[2][i]};
	double **neuron_batch[2]	 = {this->neuron[1][i], this->neuron[2][i]};
		
	for(int h = 0;h < batch_size;h++){
		derivative_batch[0][h][j] = derivative[h][j] * gamma;
		sum += derivative_batch[0][h][j] * (neuron_batch[1][h][j] - mean);
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, -1.5);
				
	sum = 0;
	for(int h = 0;h < batch_size;h++){
		sum += derivative_batch[0][h][j];
	}
	derivative_mean = -sum / sqrt(variance + epsilon);
		
	for(int h = 0;h < batch_size;h++){
		derivative_batch[1][h][j] = derivative[h][j];

		derivative[h][j] = derivative_batch[0][h][j] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_batch[1][h][j] - mean) / batch_size + derivative_mean / batch_size;
	}
}

void Multilayer_Perceptron::Resize_Memory(int batch_size){
	if(this->batch_size != batch_size){
		for(int g = 0;g < number_memory_type;g++){
			for(int i = 0;i < number_layer;i++){
				if(Access_Memory(g, i)){
					for(int h = 0;h < this->batch_size;h++){
						delete[] derivative[g][i][h];
						delete[] neuron[g][i][h];
					}
					derivative[g][i] = (double**)realloc(derivative[g][i],	sizeof(double*) * batch_size);
					neuron[g][i]	 = (double**)realloc(neuron[g][i],		sizeof(double*) * batch_size);

					for(int h = 0;h < batch_size;h++){
						derivative[g][i][h] = new double[number_neuron[i]];
						neuron[g][i][h]		= new double[number_neuron[i]];
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

Multilayer_Perceptron::Multilayer_Perceptron(char **type_layer, int number_layer, int number_neuron[]){
	this->number_layer	= number_layer;
	this->number_neuron = new int[number_layer];
	this->type_layer	= new char*[number_layer];

	batch_size			= 1;
	number_memory_type	= 3;

	for(int i = 0;i < number_layer;i++){
		this->type_layer[i] = new char[strlen(type_layer[i]) + 1];
		strcpy(this->type_layer[i], type_layer[i]);
		this->number_neuron[i] = number_neuron[i];
	}

	gamma		 = new double*[number_layer];
	beta		 = new double*[number_layer];
	mean		 = new double*[number_layer];
	variance	 = new double*[number_layer];
	sum_mean	 = new double*[number_layer];
	sum_variance = new double*[number_layer];

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			gamma[i]		= new double[number_neuron[i]];
			beta[i]			= new double[number_neuron[i]];
			mean[i]			= new double[number_neuron[i]];
			variance[i]		= new double[number_neuron[i]];
			sum_mean[i]		= new double[number_neuron[i]];
			sum_variance[i]	= new double[number_neuron[i]];
		}
	}

	derivative	= new double***[number_memory_type];
	neuron		= new double***[number_memory_type];

	for(int g = 0;g < number_memory_type;g++){
		derivative[g]	= new double**[number_layer];
		neuron[g]		= new double**[number_layer];

		for(int i = 0;i < number_layer;i++){
			if(Access_Memory(g, i)){
				derivative[g][i] = new double*[batch_size];
				neuron[g][i]	 = new double*[batch_size];

				for(int h = 0;h < batch_size;h++){
					derivative[g][i][h] = new double[number_neuron[i]];
					neuron[g][i][h]		= new double[number_neuron[i]];
				}
			}
		}
	}

	weight = new double**[number_layer];

	for(int i = 1;i < number_layer;i++){
		weight[i] = new double*[number_neuron[i]];

		for(int j = 0;j < number_neuron[i];j++){
			weight[i][j] = new double[number_neuron[i - 1] + 1];
		}
	}
}
Multilayer_Perceptron::~Multilayer_Perceptron(){
	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			delete[] gamma[i];
			delete[] beta[i];
			delete[] mean[i];
			delete[] variance[i];
			delete[] sum_mean[i];
			delete[] sum_variance[i];
		}
	}
	delete[] gamma;
	delete[] beta;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;

	for(int g = 0;g < number_memory_type;g++){
		for(int i = 0;i < number_layer;i++){
			if(Access_Memory(g, i)){
				for(int h = 0;h < batch_size;h++){
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

	for(int i = 1;i < number_layer;i++){
		for(int j = 0;j < number_neuron[i];j++){
			delete[] weight[i][j];
		}
		delete[] weight[i];
	}
	delete[] weight;

	for(int i = 0;i < number_layer;i++){
		delete[] type_layer[i];
	}
	delete[] type_layer;

	delete[] number_neuron;
}

void Multilayer_Perceptron::Initialize_Parameter(int seed, double scale, double shift){
	srand(seed);

	for(int i = 1;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_neuron[i];j++){
				gamma[i][j]	= 1;
				beta[i][j]	= 0;
			}
		}
		for(int j = 0;j < number_neuron[i];j++){
			for(int k = 0;k < number_neuron[i - 1] + 1;k++){
				weight[i][j][k] = scale * rand() / RAND_MAX + shift;
			}
		}
	}
}
void Multilayer_Perceptron::Test(double input[], double output[]){
	Resize_Memory(1);

	for(int i = 0, j = 0;j < number_neuron[i];j++){
		neuron[0][i][0][j] = input[j];
	}
	for(int i = 1;i < number_layer;i++){
		#pragma omp parallel for
		for(int j = 0;j < number_neuron[i];j++){
			Feedforward	(i, j);
			Activate	("test", i, j);
		}
		Softmax(i);
	}
	for(int i = number_layer - 1, j = 0;j < number_neuron[i];j++){
		output[j] = neuron[0][i][0][j];
	}
}

double Multilayer_Perceptron::Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output){
	int *index = new int[number_training];

	double loss = 0;

	double **target_output_batch = new double*[batch_size];

	for(int i = 0;i < number_training;i++){
		index[i] = i;
	}
	for(int i = 0;i < number_training;i++){
		int j = rand() % number_training;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;
	}

	for(int h = 0;h < batch_size;h++){
		target_output_batch[h] = new double[number_neuron[number_layer - 1]];
	}
	Resize_Memory(batch_size);

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_neuron[i];j++){
				sum_mean[i][j]		= 0;
				sum_variance[i][j]	= 0;
			}
		}
	}
	this->epsilon = epsilon;

	for(int g = 0, h = 0;g < number_training;g++){
		for(int j = 0;j < number_neuron[0];j++){
			neuron[0][0][h][j] = input[index[g]][j];
		}
		for(int j = 0;j < number_neuron[number_layer - 1];j++){
			target_output_batch[h][j] = target_output[index[g]][j];
		}

		if(++h == batch_size){
			h = 0;

			for(int i = 1;i < number_layer;i++){
				#pragma omp parallel for
				for(int j = 0;j < number_neuron[i];j++){
					Feedforward	(i, j);
					Activate	("train", i, j);
				}
				Softmax(i);
			}
			for(int i = number_layer - 1;i > 0;i--){
				#pragma omp parallel for
				for(int j = 0;j < number_neuron[i];j++){
					Backpropagate	(i, j);
					Differentiate	(i, j, learning_rate, target_output_batch);
				}
			}
			for(int i = number_layer - 1;i > 0;i--){
				#pragma omp parallel for
				for(int j = 0;j < number_neuron[i];j++){
					Adjust_Parameter(i, j);
				}
			}

			for(int h = 0;h < batch_size;h++){
				for(int i = number_layer - 1, j = 0;j < number_neuron[i];j++){
					if(strstr(type_layer[i], "ce")){
						loss -= target_output_batch[h][j] * log(neuron[0][i][h][j] + 0.000001) + (1 - target_output_batch[h][j]) * log(1 - neuron[0][i][h][j] + 0.000001);
					}
					if(strstr(type_layer[i], "mse")){
						loss += 0.5 * (neuron[0][i][h][j] - target_output_batch[h][j]) * (neuron[0][i][h][j] - target_output_batch[h][j]);
					}
				}
			}
		}
	}

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_neuron[i];j++){
				mean[i][j]		= sum_mean[i][j] / (number_training / batch_size);
				variance[i][j]	= ((double)batch_size / (batch_size - 1)) * sum_variance[i][j] / (number_training / batch_size);
			}
		}
	}

	for(int h = 0;h < batch_size;h++){
		delete[] target_output_batch[h];
	}
	delete[] index;
	delete[] target_output_batch;

	return loss / number_training;
}