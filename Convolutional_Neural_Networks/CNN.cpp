#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CNN.h"

void Convolutional_Neural_Networks::Activate(char option[], int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Activate(option, layer_index, map_index);
		}

		for(int h = 0;h < batch_size;h++){
			double mask = 1;

			if(strstr(type_layer[i], "do")){
				char *rate = strstr(type_layer[i], "do") + 2;

				if(!strcmp(option, "train")){
					mask = ((double)rand() / RAND_MAX <= atof(rate));
				}
				else
				if(!strcmp(option, "test")){
					mask = atof(rate);
				}
			}
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double &neuron = this->neuron[0][i][h][j][k][l];

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

					// dropout
					neuron *= mask;
				}
			}
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double &neuron = this->neuron[0][i][h][j][k][l];

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
	}
}
void Convolutional_Neural_Networks::Adjust_Parameter(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Adjust_Parameter(layer_index, map_index);
		}

		for(int h = 0;h < batch_size;h++){
			double ***derivative	= this->derivative[0][i][h];
			double ***lower_neuron	= this->neuron[0][i - 1][h];
		
			for(int m = 0;m < number_map[i - 1];m++){
				for(int n = 0;n < length_filter[i];n++){
					for(int o = 0;o < length_filter[i];o++){
						double sum = 0;

						for(int k = 0;k < length_map[i];k++){
							for(int l = 0;l < length_map[i];l++){
								int index[2] = {k * stride[i] + n, l * stride[i] + o};

								if(index[0] < length_map[i - 1] && index[1] < length_map[i - 1]){
									sum += derivative[j][k][l] * lower_neuron[m][index[0]][index[1]];
								}
							}
						}
						weight[i][j][m][n][o] -= sum;
					}
				}
			}

			double sum = 0;

			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					sum += derivative[j][k][l];
				}
			}
			weight[i][j][number_map[i - 1]][0][0] -= sum;
		}
	}
}
void Convolutional_Neural_Networks::Backpropagate(int layer_index, int map_index){
	if(layer_index == number_layer - 1){
		return;
	}

	int i = layer_index;
	int j = map_index;

	if(type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			double ***derivative		= this->derivative[0][i][h];
			double ***upper_derivative	= this->derivative[0][i + 1][h];

			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					int ks		 = k / stride[i + 1];
					int ls		 = l / stride[i + 1];
					int index[2] = {ks - (length_filter[i + 1] - 1), ls - (length_filter[i + 1] - 1)};

					double sum = 0;

					if(index[0] < 0) index[0] = 0;
					if(index[1] < 0) index[1] = 0;

					for(int m = 0;m < number_map[i + 1];m++){
						for(int n = index[0];n < length_map[i + 1] && n <= ks;n++){
							for(int o = index[1];o < length_map[i + 1] && o <= ls;o++){
								sum += upper_derivative[m][n][o] * weight[i + 1][m][j][abs(ks - n)][abs(ls - o)];
							}
						}
					}
					derivative[j][k][l] = sum;
				}
			}
		}
	}
	else
	if(type_layer[i + 1][0] == 'P'){
		if(strstr(type_layer[i + 1], "pad")){
			int margin = (length_map[i + 1] - length_map[i]) / 2;

			for(int h = 0;h < batch_size;h++){
				double **derivative			= this->derivative[0][i][h][j];
				double **upper_derivative	= this->derivative[0][i + 1][h][j];

				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						derivative[k][l] = upper_derivative[margin + k][margin + l];
					}
				}
			}
		}
		else{
			int stride = length_map[i] / length_map[i + 1];

			for(int h = 0;h < batch_size;h++){
				double **derivative			= this->derivative[0][i][h][j];
				double **upper_derivative	= this->derivative[0][i + 1][h][j];

				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						derivative[k][l] = upper_derivative[k / stride][l / stride];
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Differentiate(int layer_index, int map_index, double learning_rate, double **target_output){
	int i = layer_index;
	int j = map_index;

	if(type_layer[i][0] == 'C'){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double &derivative	= this->derivative[0][i][h][j][k][l];
					double &neuron		= this->neuron[0][i][h][j][k][l];

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
			}
		}

		if(strstr(type_layer[i], "bn")){
			Batch_Normalization_Differentiate(layer_index, map_index);
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double &derivative	= this->derivative[0][i][h][j][k][l];
					double &neuron		= this->neuron[0][i][h][j][k][l];

					derivative = learning_rate * (neuron - target_output[h][j]);

					if(strstr(type_layer[i], "ce")){
						if(strstr(type_layer[i], "sm")){
							// derivative = derivative;
						}
						else{
							// derivative = derivative;
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
	}
}
void Convolutional_Neural_Networks::Feedforward(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			double ***lower_neuron	= this->neuron[0][i - 1][h];
			double ***neuron		= this->neuron[0][i][h];

			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double sum = 0;

					for(int m = 0;m < number_map[i - 1];m++){
						for(int n = 0;n < length_filter[i];n++){
							for(int o = 0;o < length_filter[i];o++){
								int index[2] = {k * stride[i] + n, l * stride[i] + o};

								if(index[0] < length_map[i - 1] && index[1] < length_map[i - 1]){
									sum += lower_neuron[m][index[0]][index[1]] * weight[i][j][m][n][o];
								}
							}
						}
					}
					neuron[j][k][l] = sum + weight[i][j][number_map[i - 1]][0][0];
				}
			}
		}
	}
	else
	if(type_layer[i][0] == 'P'){
		if(strstr(type_layer[i], "pad")){
			int margin = (length_map[i] - length_map[i - 1]) / 2;

			for(int h = 0;h < batch_size;h++){
				double **lower_neuron	= this->neuron[0][i - 1][h][j];
				double **neuron			= this->neuron[0][i][h][j];

				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						neuron[k][l] = 0;
					}
				}
				for(int k = 0;k < length_map[i - 1];k++){
					for(int l = 0;l < length_map[i - 1];l++){
						neuron[margin + k][margin + l] = lower_neuron[k][l];
					}
				}
			}
		}
		else{
			int stride = length_map[i - 1] / length_map[i];

			for(int h = 0;h < batch_size;h++){
				double **lower_neuron	= this->neuron[0][i - 1][h][j];
				double **neuron			= this->neuron[0][i][h][j];

				for(int k = 0;k < length_map[i];k++){
					for(int l = 0;l < length_map[i];l++){
						if(strstr(type_layer[i], "avg")){
							double sum = 0;
						
							for(int m = 0;m < stride;m++){
								for(int n = 0;n < stride;n++){
									sum += lower_neuron[k * stride + m][l * stride + n];
								}
							}
							neuron[k][l] = sum / (stride * stride);
						}
						else
						if(strstr(type_layer[i], "max")){
							double max = -1;
						
							for(int m = 0;m < stride;m++){
								for(int n = 0;n < stride;n++){
									if(max < lower_neuron[k * stride + m][l * stride + n]){
										max = lower_neuron[k * stride + m][l * stride + n];
									}
								}
							}
							neuron[k][l] = max;
						}
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Softmax(int layer_index){
	int i = layer_index;

	if(strstr(type_layer[i], "sm")){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					double max = 0;
					double sum = 0;

					double ***neuron = this->neuron[0][i][h];

					for(int j = 0;j < number_map[i];j++){
						if(max < neuron[j][k][l]){
							max = neuron[j][k][l];
						}
					}
					for(int j = 0;j < number_map[i];j++){
						neuron[j][k][l] = exp(neuron[j][k][l] - max);
						sum += neuron[j][k][l];
					}
					for(int j = 0;j < number_map[i];j++){
						neuron[j][k][l] /= sum;
					}
				}
			}
		}
	}
}

void Convolutional_Neural_Networks::Batch_Normalization_Activate(char option[], int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	double gamma		 = this->gamma[i][j];
	double beta			 = this->beta[i][j];
	double &mean		 = this->mean[i][j];
	double &variance	 = this->variance[i][j];
	double &sum_mean	 = this->sum_mean[i][j];
	double &sum_variance = this->sum_variance[i][j];

	double ****neuron			= this->neuron[0][i];
	double ****neuron_batch[2]	= {this->neuron[1][i], this->neuron[2][i]};

	if(!strcmp(option, "train")){
		double sum = 0;

		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					sum += neuron[h][j][k][l];
				}
			}
		}
		sum_mean += (mean = sum / (batch_size * length_map[i] * length_map[i]));
							
		sum = 0;
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					sum += (neuron[h][j][k][l] - mean) * (neuron[h][j][k][l] - mean);
				}
			}
		}
		sum_variance += (variance = sum / (batch_size * length_map[i] * length_map[i]));
			
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					neuron_batch[0][h][j][k][l] = (neuron[h][j][k][l] - mean) / sqrt(variance + epsilon);
					neuron_batch[1][h][j][k][l] = neuron[h][j][k][l];

					neuron[h][j][k][l] = gamma * neuron_batch[0][h][j][k][l] + beta;
				}
			}
		}
	}
	else
	if(!strcmp(option, "test")){
		double stdv = sqrt(variance + epsilon);

		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < length_map[i];k++){
				for(int l = 0;l < length_map[i];l++){
					neuron[h][j][k][l] = gamma / stdv * neuron[h][j][k][l] + (beta - gamma * mean / stdv);
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Batch_Normalization_Adjust_Parameter(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	double sum = 0;

	double ****derivative_batch	= this->derivative[2][i];
	double ****neuron_batch		= this->neuron[1][i];
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < length_map[i];k++){
			for(int l = 0;l < length_map[i];l++){
				sum += derivative_batch[h][j][k][l] * neuron_batch[h][j][k][l];
			}
		}
	}
	gamma[i][j] -= sum;
						
	sum = 0;
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < length_map[i];k++){
			for(int l = 0;l < length_map[i];l++){
				sum += derivative_batch[h][j][k][l];
			}
		}
	}
	beta[i][j] -= sum;
}
void Convolutional_Neural_Networks::Batch_Normalization_Differentiate(int layer_index, int map_index){
	int i = layer_index;
	int j = map_index;

	double derivative_mean;
	double derivative_variance;
	double sum = 0;

	double gamma	= this->gamma[i][j];
	double beta		= this->beta[i][j];
	double mean		= this->mean[i][j];
	double variance	= this->variance[i][j];

	double ****derivative			= this->derivative[0][i];
	double ****derivative_batch[2]	= {this->derivative[1][i], this->derivative[2][i]};
	double ****neuron_batch[2]		= {this->neuron[1][i], this->neuron[2][i]};
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < length_map[i];k++){
			for(int l = 0;l < length_map[i];l++){
				derivative_batch[0][h][j][k][l] = derivative[h][j][k][l] * gamma;
				sum += derivative_batch[0][h][j][k][l] * (neuron_batch[1][h][j][k][l] - mean);
			}
		}
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, -1.5);
				
	sum = 0;
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < length_map[i];k++){
			for(int l = 0;l < length_map[i];l++){
				sum += derivative_batch[0][h][j][k][l];
			}
		}
	}
	derivative_mean = -sum / sqrt(variance + epsilon);
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < length_map[i];k++){
			for(int l = 0;l < length_map[i];l++){
				derivative_batch[1][h][j][k][l] = derivative[h][j][k][l];

				derivative[h][j][k][l] = derivative_batch[0][h][j][k][l] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_batch[1][h][j][k][l] - mean) / (batch_size * length_map[i] * length_map[i]) + derivative_mean / (batch_size * length_map[i] * length_map[i]);
			}
		}
	}
}

void Convolutional_Neural_Networks::Resize_Memory(int batch_size){
	if(this->batch_size != batch_size){
		for(int g = 0;g < number_memory_type;g++){
			for(int i = 0;i < number_layer;i++){
				if(Access_Memory(g, i)){
					for(int h = 0;h < this->batch_size;h++){
						for(int j = 0;j < number_map[i];j++){
							for(int k = 0;k < length_map[i];k++){
								delete[] derivative[g][i][h][j][k];
								delete[] neuron[g][i][h][j][k];
							}
							delete[] derivative[g][i][h][j];
							delete[] neuron[g][i][h][j];
						}
						delete[] derivative[g][i][h];
						delete[] neuron[g][i][h];
					}
					derivative[g][i] = (double****)realloc(derivative[g][i], sizeof(double***) * batch_size);
					neuron[g][i]	 = (double****)realloc(neuron[g][i],	 sizeof(double***) * batch_size);

					for(int h = 0;h < batch_size;h++){
						derivative[g][i][h] = new double**[number_map[i]];
						neuron[g][i][h]		= new double**[number_map[i]];

						for(int j = 0;j < number_map[i];j++){
							derivative[g][i][h][j]	= new double*[length_map[i]];
							neuron[g][i][h][j]		= new double*[length_map[i]];

							for(int k = 0;k < length_map[i];k++){
								derivative[g][i][h][j][k]	= new double[length_map[i]];
								neuron[g][i][h][j][k]		= new double[length_map[i]];
							}
						}
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

Convolutional_Neural_Networks::Convolutional_Neural_Networks(char **type_layer, int number_layer, int length_map[], int number_map[]){
	this->length_filter	= new int[number_layer];
	this->length_map	= new int[number_layer];
	this->number_layer	= number_layer;
	this->number_map	= new int[number_layer];
	this->stride		= new int[number_layer];
	this->type_layer	= new char*[number_layer];

	batch_size			= 1;
	number_memory_type	= 3;

	for(int i = 0;i < number_layer;i++){
		this->type_layer[i] = new char[strlen(type_layer[i]) + 1];
		strcpy(this->type_layer[i], type_layer[i]);
		this->number_map[i] = number_map[i];
		this->length_map[i] = (length_map == 0) ? (1):(length_map[i]);

		if(strstr(type_layer[i], "fs"))	length_filter[i] = atoi(strstr(type_layer[i], "fs") + 2);
		else							length_filter[i] = (i == 0 || type_layer[i][0] == 'P') ? (0):(this->length_map[i - 1] - this->length_map[i] + 1);

		if(strstr(type_layer[i], "st"))	stride[i] = atoi(strstr(type_layer[i], "st") + 2);
		else							stride[i] = 1;
	}

	gamma		 = new double*[number_layer];
	beta		 = new double*[number_layer];
	mean		 = new double*[number_layer];
	variance	 = new double*[number_layer];
	sum_mean	 = new double*[number_layer];
	sum_variance = new double*[number_layer];

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			gamma[i]		= new double[number_map[i]];
			beta[i]			= new double[number_map[i]];
			mean[i]			= new double[number_map[i]];
			variance[i]		= new double[number_map[i]];
			sum_mean[i]		= new double[number_map[i]];
			sum_variance[i]	= new double[number_map[i]];
		}
	}

	derivative	= new double*****[number_memory_type];
	neuron		= new double*****[number_memory_type];

	for(int g = 0;g < number_memory_type;g++){
		derivative[g]	= new double****[number_layer];
		neuron[g]		= new double****[number_layer];

		for(int i = 0;i < number_layer;i++){
			if(Access_Memory(g, i)){
				derivative[g][i] = new double***[batch_size];
				neuron[g][i]	 = new double***[batch_size];

				for(int h = 0;h < batch_size;h++){
					derivative[g][i][h]	= new double**[number_map[i]];
					neuron[g][i][h]		= new double**[number_map[i]];

					for(int j = 0;j < number_map[i];j++){
						derivative[g][i][h][j]	= new double*[length_map[i]];
						neuron[g][i][h][j]		= new double*[length_map[i]];

						for(int k = 0;k < length_map[i];k++){
							derivative[g][i][h][j][k]	= new double[length_map[i]];
							neuron[g][i][h][j][k]		= new double[length_map[i]];
						}
					}
				}
			}
		}
	}

	weight = new double****[number_layer];

	for(int i = 0;i < number_layer;i++){
		if(length_filter[i] > 0){
			weight[i] = new double***[number_map[i]];

			for(int j = 0;j < number_map[i];j++){
				weight[i][j] = new double**[number_map[i - 1] + 1];

				for(int k = 0;k < number_map[i - 1] + 1;k++){
					weight[i][j][k] = new double*[length_filter[i]];

					for(int l = 0;l < length_filter[i];l++){
						weight[i][j][k][l] = new double[length_filter[i]];
					}
				}
			}
		}
	}
}
Convolutional_Neural_Networks::~Convolutional_Neural_Networks(){
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
					for(int j = 0;j < number_map[i];j++){
						for(int k = 0;k < length_map[i];k++){
							delete[] derivative[g][i][h][j][k];
							delete[] neuron[g][i][h][j][k];
						}
						delete[] derivative[g][i][h][j];
						delete[] neuron[g][i][h][j];
					}
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

	for(int i = 0;i < number_layer;i++){
		if(length_filter[i] > 0){
			for(int j = 0;j < number_map[i];j++){
				for(int k = 0;k < number_map[i - 1] + 1;k++){
					for(int l = 0;l < length_filter[i];l++){
						delete[] weight[i][j][k][l];
					}
					delete[] weight[i][j][k];
				}
				delete[] weight[i][j];
			}
			delete[] weight[i];
		}
	}
	delete[] weight;

	for(int i = 0;i < number_layer;i++){
		delete[] type_layer[i];
	}
	delete[] type_layer;

	delete[] length_filter;
	delete[] length_map;
	delete[] number_map;
	delete[] stride;
}

void Convolutional_Neural_Networks::Initialize_Parameter(int seed, double scale, double shift){
	srand(seed);

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_map[i];j++){
				gamma[i][j]	= 1;
				beta[i][j]	= 0;
			}
		}
		if(length_filter[i] > 0){
			for(int j = 0;j < number_map[i];j++){
				for(int k = 0;k < number_map[i - 1] + 1;k++){
					for(int l = 0;l < length_filter[i];l++){
						for(int m = 0;m < length_filter[i];m++){
							weight[i][j][k][l][m] = scale * rand() / RAND_MAX + shift;
						}
					}
				}
			}
		}
	}
}
void Convolutional_Neural_Networks::Load_Parameter(char path[]){
	FILE *file = fopen(path, "rt");

	if(file){
		fscanf(file, "%lf", &epsilon);

		for(int i = 0;i < number_layer;i++){
			if(strstr(type_layer[i], "bn")){
				for(int j = 0;j < number_map[i];j++) fscanf(file, "%lf", &gamma[i][j]);
				for(int j = 0;j < number_map[i];j++) fscanf(file, "%lf", &beta[i][j]);
				for(int j = 0;j < number_map[i];j++) fscanf(file, "%lf", &mean[i][j]);
				for(int j = 0;j < number_map[i];j++) fscanf(file, "%lf", &variance[i][j]);
			}
			if(length_filter[i] > 0){
				for(int j = 0;j < number_map[i];j++){
					for(int k = 0;k < number_map[i - 1] + 1;k++){
						for(int l = 0;l < length_filter[i];l++){
							for(int m = 0;m < length_filter[i];m++){
								fscanf(file, "%lf", &weight[i][j][k][l][m]);
							}
						}
					}
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load_Parameter], %s not found\n", path);
	}
}
void Convolutional_Neural_Networks::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	fprintf(file, "%f\n", epsilon);

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_map[i];j++) fprintf(file, "%f\n", gamma[i][j]);
			for(int j = 0;j < number_map[i];j++) fprintf(file, "%f\n", beta[i][j]);
			for(int j = 0;j < number_map[i];j++) fprintf(file, "%f\n", mean[i][j]);
			for(int j = 0;j < number_map[i];j++) fprintf(file, "%f\n", variance[i][j]);
		}
		if(length_filter[i] > 0){
			for(int j = 0;j < number_map[i];j++){
				for(int k = 0;k < number_map[i - 1] + 1;k++){
					for(int l = 0;l < length_filter[i];l++){
						for(int m = 0;m < length_filter[i];m++){
							fprintf(file, "%f\n", weight[i][j][k][l][m]);
						}
					}
				}
			}
		}
	}
	fclose(file);
}
void Convolutional_Neural_Networks::Test(double input[], double output[]){
	Resize_Memory(1);

	#pragma omp parallel for
	for(int h = 0;h < number_map[0] * length_map[0] * length_map[0];h++){
		int j = (h / (length_map[0] * length_map[0]));
		int k = (h % (length_map[0] * length_map[0])) / length_map[0];
		int l = (h % (length_map[0] * length_map[0])) % length_map[0];

		neuron[0][0][0][j][k][l] = input[h];
	}

	for(int i = 1;i < number_layer;i++){
		#pragma omp parallel for
		for(int j = 0;j < number_map[i];j++){
			Feedforward	(i, j);
			Activate	("test", i, j);
		}
		Softmax(i);
	}
	for(int i = number_layer - 1, j = 0;j < number_map[i];j++){
		output[j] = neuron[0][i][0][j][0][0];
	}
}

double Convolutional_Neural_Networks::Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output){
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
		target_output_batch[h] = new double[number_map[number_layer - 1]];
	}
	Resize_Memory(batch_size);

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_map[i];j++){
				sum_mean[i][j]		= 0;
				sum_variance[i][j]	= 0;
			}
		}
	}
	this->epsilon = epsilon;

	for(int g = 0, h = 0;g < number_training;g++){
		#pragma omp parallel for
		for(int i = 0;i < number_map[0] * length_map[0] * length_map[0];i++){
			int j = (i / (length_map[0] * length_map[0]));
			int k = (i % (length_map[0] * length_map[0])) / length_map[0];
			int l = (i % (length_map[0] * length_map[0])) % length_map[0];

			neuron[0][0][h][j][k][l] = input[index[g]][i];
		}
		for(int j = 0;j < number_map[number_layer - 1];j++){
			target_output_batch[h][j] = target_output[index[g]][j];
		}

		if(++h == batch_size){
			h = 0;

			for(int i = 1;i < number_layer;i++){
				#pragma omp parallel for
				for(int j = 0;j < number_map[i];j++){
					Feedforward	(i, j);
					Activate	("train", i, j);
				}
				Softmax(i);
			}

			for(int i = number_layer - 1;i > 0;i--){
				#pragma omp parallel for
				for(int j = 0;j < number_map[i];j++){
					Backpropagate(i, j);
					Differentiate(i, j, learning_rate, target_output_batch);
				}
			}
			for(int i = number_layer - 1;i > 0;i--){
				#pragma omp parallel for
				for(int j = 0;j < number_map[i];j++){
					Adjust_Parameter(i, j);
				}
			}

			for(int h = 0;h < batch_size;h++){
				for(int i = number_layer - 1, j = 0;j < number_map[i];j++){
					if(strstr(type_layer[i], "ce")){
						loss -= target_output_batch[h][j] * log(neuron[0][i][h][j][0][0] + 0.000001) + (1 - target_output_batch[h][j]) * log(1 - neuron[0][i][h][j][0][0] + 0.000001);
					}
					if(strstr(type_layer[i], "mse")){
						loss += 0.5 * (neuron[0][i][h][j][0][0] - target_output_batch[h][j]) * (neuron[0][i][h][j][0][0] - target_output_batch[h][j]);
					}
				}
			}
		}
	}

	for(int i = 0;i < number_layer;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_map[i];j++){
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
