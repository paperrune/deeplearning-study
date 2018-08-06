#include <fstream>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <unordered_map>

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "Neural_Networks.h"

#define NUMBER_THREADS 64

__device__ void Activate(Layer &layer, int j) {
	float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];

	if (layer.activation == Activation::softmax) {
		if (j == 0) {
			double max;
			double sum = 0;

			for (int j = 0; j < layer.number_nodes; j++) {
				if (j == 0 || max < neuron[j]) {
					max = neuron[j];
				}
			}
			for (int j = 0; j < layer.number_nodes; j++) {
				sum += (neuron[j] = exp(neuron[j] - max));
			}
			for (int j = 0; j < layer.number_nodes; j++) {
				neuron[j] /= sum;
			}
		}
	}
	else if (layer.activation == Activation::linear) {
		// neuron[j] = neuron[j];
	}
	else if (layer.activation == Activation::relu) {
		neuron[j] *= (neuron[j] > 0);
	}
	else if (layer.activation == Activation::sigmoid) {
		neuron[j] = 1 / (1 + exp(-neuron[j]));
	}
	else if (layer.activation == Activation::tanh) {
		neuron[j] = 2 / (1 + exp(-2 * neuron[j])) - 1;
	}
}
__device__ void Differentiate(Layer &layer, int index, int loss, float y_data[], double scale = 1) {
	if (y_data) {
		if (loss == Loss::cross_entropy) {
			if (layer.activation == Activation::sigmoid) {
				layer.error[index] = (layer.neuron[index] - y_data[index]) / (layer.number_nodes * layer.batch_size);
			}
			else if (layer.activation == Activation::softmax) {
				layer.error[index] = (layer.neuron[index] - y_data[index]) / layer.batch_size;
			}
		}
		else if (loss == Loss::mean_squared_error) {
			layer.error[index] = 2 * (layer.neuron[index] - y_data[index]) / (layer.number_nodes * layer.batch_size);
		}
	}
	else {
		double neuron = layer.neuron[index] * scale;

		if (layer.activation == Activation::linear) {
			// error[index] *= 1;
		}
		else if (layer.activation == Activation::relu) {
			layer.error[index] = (layer.neuron[index] > 0) ? (layer.error[index]) : (0);
		}
		else if (layer.activation == Activation::sigmoid && loss != Loss::cross_entropy) {
			layer.error[index] *= (1 - neuron) * neuron;
		}
		else if (layer.activation == Activation::tanh) {
			layer.error[index] *= (1 - neuron) * (1 + neuron);
		}
	}
}

__device__ double Calculate_Gradient(Optimizer &optimizer, int index, double gradient, int iterations) {
	double learning_rate = optimizer.learning_rate / (1 + optimizer.decay * iterations);
	double output = 0;

	if (optimizer.type == 0) { // SGD
		output = -learning_rate * gradient;
	}
	else if (optimizer.type == 1) { // Momentum
		output = optimizer.gradient[index] * optimizer.momentum[0] - learning_rate * gradient;
		optimizer.gradient[index] = output;
	}
	else if (optimizer.type == 2) { // Nesterov
		output = (optimizer.gradient[index] * optimizer.momentum[0] - learning_rate * gradient) * optimizer.momentum[0] - learning_rate * gradient;
		optimizer.gradient[index] = output;
	}
	else if (optimizer.type == 3) { // Adam
		double m1 = optimizer.momentum[0] * optimizer.memory[0][index] + (1 - optimizer.momentum[0]) * gradient;
		double m2 = optimizer.momentum[1] * optimizer.memory[1][index] + (1 - optimizer.momentum[1]) * gradient * gradient;

		optimizer.memory[0][index] = m1;
		optimizer.memory[1][index] = m2;
		m1 /= (1 - optimizer.momentum[0]);
		m2 /= (1 - optimizer.momentum[1]);
		output = -learning_rate * m1 / sqrt(m2 + optimizer.epsilon);
	}
	return output;
}

__global__ void Activate(Batch_Normalization batch_normalization, float _neuron[], bool training) {
	int j = blockIdx.x;

	if (training) {
		float *neuron = &_neuron[j * batch_normalization.map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[j * batch_normalization.map_size];
		float *neuron_normalized = &batch_normalization.neuron_normalized[j * batch_normalization.map_size];

		__shared__ double standard_deviation;
		__shared__ double sum[NUMBER_THREADS];

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				sum[threadIdx.x] += neuron[index + k];
			}
		}
		for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
			__syncthreads();

			if (threadIdx.x < h) {
				sum[threadIdx.x] += sum[threadIdx.x + h];
			}
		}
		if (threadIdx.x == 0) {
			batch_normalization.moving_mean[j] = batch_normalization.momentum * batch_normalization.moving_mean[j] + (1 - batch_normalization.momentum) * (batch_normalization.mean[j] = sum[0] / (batch_normalization.batch_size * batch_normalization.map_size));
		}

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				sum[threadIdx.x] += (neuron[index + k] - batch_normalization.mean[j]) * (neuron[index + k] - batch_normalization.mean[j]);
			}
		}
		for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
			__syncthreads();

			if (threadIdx.x < h) {
				sum[threadIdx.x] += sum[threadIdx.x + h];
			}
		}
		if (threadIdx.x == 0) {
			batch_normalization.moving_variance[j] = batch_normalization.momentum * batch_normalization.moving_variance[j] + (1 - batch_normalization.momentum) * (batch_normalization.variance[j] = sum[0] / (batch_normalization.batch_size * batch_normalization.map_size));
			standard_deviation = sqrt(batch_normalization.variance[j] + batch_normalization.epsilon);
		}
		__syncthreads();

		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				neuron_backup[index + k] = neuron[index + k];
				neuron_normalized[index + k] = (neuron[index + k] - batch_normalization.mean[j]) / standard_deviation;
				neuron[index + k] = batch_normalization.gamma[j] * neuron_normalized[index + k] + batch_normalization.beta[j];
			}
		}
	}
	else {
		float *neuron = &_neuron[j * batch_normalization.map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[j * batch_normalization.map_size];

		double standard_deviation = sqrt(batch_normalization.moving_variance[j] + batch_normalization.epsilon);

		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				neuron_backup[index + k] = neuron[index + k];
				neuron[index + k] = batch_normalization.gamma[j] / standard_deviation * neuron[index + k] + (batch_normalization.beta[j] - batch_normalization.gamma[j] * batch_normalization.moving_mean[j] / standard_deviation);
			}
		}
	}
}
__global__ void Activate(Layer layer) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		int index = blockIdx.x * layer.number_nodes + j;

		if (layer.bias) {
			layer.neuron[index] += layer.bias[j / layer.map_size];
		}
		Activate(layer, j);
	}
}
__global__ void Activate(Layer layer, ::Dropout dropout, bool training) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		int index = blockIdx.x * layer.number_nodes + j;

		if (layer.bias) {
			layer.neuron[index] += layer.bias[j / layer.map_size];
		}
		if (training){
			if (dropout.mask[index]) {
				layer.neuron[index] /= (1 - dropout.rate);
			}
			else {
				layer.neuron[index] = 0;
				return;
			}
		}
		Activate(layer, j);
	}
}
__global__ void Adjust_Bias(Layer layer, Optimizer optimizer, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_maps) {
		double sum = 0;

		for (int h = 0; h < layer.batch_size; h++) {
			for (int k = 0; k < layer.map_size; k++) {
				sum += layer.error[h * layer.number_nodes + j * layer.map_size + k];
			}
		}
		layer.bias[j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Parameter(Batch_Normalization batch_normalization, Optimizer gamma_optimizer, Optimizer beta_optimizer, int iterations) {
	int j = blockIdx.x;

	float *error_backup = &batch_normalization.error_backup[j * batch_normalization.map_size];
	float *neuron_normalized = &batch_normalization.neuron_normalized[j * batch_normalization.map_size];

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			sum[threadIdx.x] += error_backup[index + k] * neuron_normalized[index + k];
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		batch_normalization.gamma[j] += Calculate_Gradient(gamma_optimizer, j, sum[0], iterations);
	}

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			sum[threadIdx.x] += error_backup[index + k];
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		batch_normalization.beta[j] += Calculate_Gradient(beta_optimizer, j, sum[0], iterations);
	}
}
__global__ void Adjust_Weight(Layer layer, Layer parent_layer, Connection connection, Optimizer optimizer, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < connection.number_weights) {
		double sum = 0;

		for (int h = 0; h < layer.batch_size; h++) {
			float *error = &layer.error[h * layer.number_nodes];
			float *prev_neuron = &parent_layer.neuron[h * parent_layer.number_nodes];

			for (Index *from_weight = &connection.from_weight[j]; from_weight->weight != -1; from_weight += connection.number_weights) {
				sum += error[from_weight->next_node] * prev_neuron[from_weight->prev_node];
			}
		}
		connection.weight[j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Backward(Layer layer, Layer parent_layer, Connection connection, int type) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < parent_layer.number_nodes) {
		float *error = &layer.error[blockIdx.x * layer.number_nodes];
		float *prev_error = &parent_layer.error[blockIdx.x * parent_layer.number_nodes];

		double sum = 0;

		if (type == 0) { // average-pooling
			for (Index *from_error = &connection.from_error[j]; from_error->weight != -1; from_error += parent_layer.number_nodes) {
				sum += error[from_error->next_node] / connection.from_neuron[from_error->next_node].weight;
			}
			prev_error[j] += sum;
			return;
		}
		if (type == 1) { // max-pooling
			float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];
			float *prev_neuron = &parent_layer.neuron[blockIdx.x * parent_layer.number_nodes];

			for (Index *from_error = &connection.from_error[j]; from_error->weight != -1; from_error += parent_layer.number_nodes) {
				if (prev_neuron[j] == neuron[from_error->next_node]) {
					sum += error[from_error->next_node];
				}
			}
			prev_error[j] += sum;
			return;
		}
		if (type == 2) {
			for (Index *from_error = &connection.from_error[j]; from_error->weight != -1; from_error += parent_layer.number_nodes) {
				sum += error[from_error->next_node] * connection.weight[from_error->weight];
			}
			prev_error[j] += sum;
		}
	}
}
__global__ void Calculate_Loss(Layer layer, int loss, float y_data[]) {
	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;

	for (int j = threadIdx.x; j < layer.batch_size * layer.number_nodes; j += blockDim.x) {
		if (loss == Loss::cross_entropy) {
			if (layer.activation == Activation::sigmoid) {
				sum[threadIdx.x] -= (y_data[j] * log(layer.neuron[j] + 0.00000001) + (1 - y_data[j]) * log(1.00000001 - layer.neuron[j])) / layer.number_nodes;
			}
			else if (layer.activation == Activation::softmax) {
				sum[threadIdx.x] -= y_data[j] * log(layer.neuron[j] + 0.00000001);
			}
		}
		else if (loss == Loss::mean_squared_error) {
			sum[threadIdx.x] += (layer.neuron[j] - y_data[j]) * (layer.neuron[j] - y_data[j]) / layer.number_nodes;
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		y_data[0] = sum[0];
	}
}
__global__ void Differentiate(Batch_Normalization batch_normalization, float _error[]) {
	int j = blockIdx.x;

	float *error = &_error[j * batch_normalization.map_size];
	float *error_backup = &batch_normalization.error_backup[j * batch_normalization.map_size];
	float *error_normalized = &batch_normalization.error_normalized[j * batch_normalization.map_size];
	float *neuron_backup = &batch_normalization.neuron_backup[j * batch_normalization.map_size];

	double standard_deviation = sqrt(batch_normalization.variance[j] + batch_normalization.epsilon);

	__shared__ double error_mean;
	__shared__ double error_variance;
	__shared__ double sum[2][NUMBER_THREADS];

	sum[0][threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			error_normalized[index + k] = error[index + k] * batch_normalization.gamma[j];
			sum[0][threadIdx.x] += error_normalized[index + k] * (neuron_backup[index + k] - batch_normalization.mean[j]);
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[0][threadIdx.x] += sum[0][threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		error_variance = sum[0][threadIdx.x] * (-0.5) * pow(batch_normalization.variance[j] + batch_normalization.epsilon, -1.5);
	}

	sum[0][threadIdx.x] = 0;
	sum[1][threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			sum[0][threadIdx.x] += error_normalized[index + k];
			sum[1][threadIdx.x] += (neuron_backup[index + k] - batch_normalization.mean[j]);
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[0][threadIdx.x] += sum[0][threadIdx.x + h];
			sum[1][threadIdx.x] += sum[1][threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		error_mean = -sum[0][threadIdx.x] / standard_deviation + error_variance * (-2) * sum[1][threadIdx.x] / (batch_normalization.batch_size * batch_normalization.map_size);
	}
	__syncthreads();

	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			error_backup[index + k] = error[index + k];
			error[index + k] = error_normalized[index + k] / standard_deviation + error_variance * 2 * (neuron_backup[index + k] - batch_normalization.mean[j]) / (batch_normalization.batch_size * batch_normalization.map_size) + error_mean / (batch_normalization.batch_size * batch_normalization.map_size);
		}
	}
}
__global__ void Differentiate(Layer layer, int loss = -1, float y_data[] = nullptr) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		Differentiate(layer, blockIdx.x * layer.number_nodes + j, loss, y_data);
	}
}
__global__ void Differentiate(Layer layer, Dropout dropout, int loss = -1, float y_data[] = nullptr) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		int index = blockIdx.x * layer.number_nodes + j;

		if (dropout.mask[index]) {
			Differentiate(layer, index, loss, y_data, dropout.rate);
			layer.error[index] /= (1 - dropout.rate);
		}
		else {
			layer.error[index] = 0;
		}
	}
}
__global__ void Forward(Layer layer, Layer parent_layer, Connection connection, int type) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];
		float *prev_neuron = &parent_layer.neuron[blockIdx.x * parent_layer.number_nodes];

		double sum = 0;

		if (type == 0) { // average-pooling
			int number_connections = 0;

			for (Index *from_neuron = &connection.from_neuron[j]; from_neuron->weight != -1; from_neuron += layer.number_nodes, number_connections++) {
				sum += prev_neuron[from_neuron->prev_node];
			}
			neuron[j] += sum / number_connections;
			return;
		}
		if (type == 1) { // max-pooling
			for (Index *from_neuron = &connection.from_neuron[j]; from_neuron->weight != -1; from_neuron += layer.number_nodes) {
				if (from_neuron == &connection.from_neuron[j] || sum < prev_neuron[from_neuron->prev_node]) {
					sum = prev_neuron[from_neuron->prev_node];
				}
			}
			neuron[j] += sum;
			return;
		}
		if (type == 2) {
			for (Index *from_neuron = &connection.from_neuron[j]; from_neuron->weight != -1; from_neuron += layer.number_nodes) {
				sum += prev_neuron[from_neuron->prev_node] * connection.weight[from_neuron->weight];
			}
			neuron[j] += sum;
		}
	}
}

Batch_Normalization::Batch_Normalization(int number_maps, int map_size, double epsilon, double momentum, Layer *layer) {
	this->batch_size = 1;
	this->beta_initializer = new ::Initializer(0);
	this->beta_optimizer = nullptr;
	this->epsilon = epsilon;
	this->gamma_initializer = new ::Initializer(1);
	this->gamma_optimizer = nullptr;
	this->layer = layer;
	this->map_size = map_size;
	this->momentum = momentum;
	this->moving_mean_initializer = new ::Initializer(0);
	this->moving_variance_initializer = new ::Initializer(1);
	this->number_maps = number_maps;
	this->number_nodes = number_maps * map_size;

	cudaMalloc(&beta, sizeof(float) * number_maps);
	cudaMalloc(&gamma, sizeof(float) * number_maps);
	cudaMalloc(&mean, sizeof(float) * number_maps);
	cudaMalloc(&variance, sizeof(float) * number_maps);
	cudaMalloc(&moving_mean, sizeof(float) * number_maps);
	cudaMalloc(&moving_variance, sizeof(float) * number_maps);

	cudaMalloc(&error_backup, sizeof(float) * number_nodes);
	cudaMalloc(&error_normalized, sizeof(float) * number_nodes);
	cudaMalloc(&neuron_backup, sizeof(float) * number_nodes);
	cudaMalloc(&neuron_normalized, sizeof(float) * number_nodes);
}
Batch_Normalization::~Batch_Normalization() {}

void Batch_Normalization::Activate(float neuron[], bool training) {
	::Activate << <number_maps, NUMBER_THREADS >> > (*this, neuron, training);
}
void Batch_Normalization::Adjust_Parameter(int iterations) {
	::Adjust_Parameter << <number_maps, NUMBER_THREADS >> > (*this, *gamma_optimizer, *beta_optimizer, iterations);
}
void Batch_Normalization::Destruct() {
	if (beta_optimizer) {
		beta_optimizer->Destruct();
		delete beta_optimizer;
	}
	if (gamma_optimizer) {
		gamma_optimizer->Destruct();
		delete gamma_optimizer;
	}
	cudaFree(beta);
	cudaFree(gamma);
	cudaFree(mean);
	cudaFree(variance);
	cudaFree(moving_mean);
	cudaFree(moving_variance);

	cudaFree(error_backup);
	cudaFree(error_normalized);
	cudaFree(neuron_backup);
	cudaFree(neuron_normalized);

	delete beta_initializer;
	delete gamma_initializer;
	delete moving_mean_initializer;
	delete moving_variance_initializer;
}
void Batch_Normalization::Differentiate(float error[]) {
	::Differentiate << <number_maps, NUMBER_THREADS >> > (*this, error);
}
void Batch_Normalization::Initialize() {
	beta_initializer->Random(number_maps, beta, number_maps, number_maps);
	gamma_initializer->Random(number_maps, gamma, number_maps, number_maps);
	moving_mean_initializer->Random(number_maps, moving_mean, number_maps, number_maps);
	moving_variance_initializer->Random(number_maps, moving_variance, number_maps, number_maps);
}
void Batch_Normalization::Optimizer(::Optimizer &optimizer) {
	if (beta_optimizer) {
		beta_optimizer->Destruct();
		delete beta_optimizer;
	}
	if (gamma_optimizer) {
		gamma_optimizer->Destruct();
		delete gamma_optimizer;
	}
	beta_optimizer = optimizer.Copy(number_maps);
	gamma_optimizer = optimizer.Copy(number_maps);
}
void Batch_Normalization::Resize_Memory(int batch_size) {
	int memory_size = sizeof(float) * batch_size * number_nodes;

	if (this->batch_size != batch_size) {
		cudaFree(error_backup);
		cudaFree(error_normalized);
		cudaFree(neuron_backup);
		cudaFree(neuron_normalized);

		cudaMalloc(&error_backup, memory_size);
		cudaMalloc(&error_normalized, memory_size);
		cudaMalloc(&neuron_backup, memory_size);
		cudaMalloc(&neuron_normalized, memory_size);

		this->batch_size = batch_size;
	}
	cudaMemset(error_backup, 0, memory_size);
	cudaMemset(error_normalized, 0, memory_size);
	cudaMemset(neuron_backup, 0, memory_size);
	cudaMemset(neuron_normalized, 0, memory_size);
}

Batch_Normalization* Batch_Normalization::Beta_Initializer(Initializer initializer) {
	if (beta_initializer) {
		delete beta_initializer;
	}
	beta_initializer = initializer.Copy();
	return this;
}
Batch_Normalization* Batch_Normalization::Copy() {
	Batch_Normalization *batch_normalization = new Batch_Normalization(number_maps, map_size, epsilon, momentum, layer);

	batch_normalization->Beta_Initializer(*beta_initializer);
	batch_normalization->Gamma_Initializer(*gamma_initializer);
	batch_normalization->Moving_Mean_Initializer(*moving_mean_initializer);
	batch_normalization->Moving_Variance_Initializer(*moving_variance_initializer);
	batch_normalization->Optimizer(*gamma_optimizer);
	batch_normalization->Resize_Memory(batch_size);
	return batch_normalization;
}
Batch_Normalization* Batch_Normalization::Gamma_Initializer(Initializer initializer) {
	if (gamma_initializer) {
		delete gamma_initializer;
	}
	gamma_initializer = initializer.Copy();
	return this;
}
Batch_Normalization* Batch_Normalization::Moving_Mean_Initializer(Initializer initializer) {
	if (moving_mean_initializer) {
		delete moving_mean_initializer;
	}
	moving_mean_initializer = initializer.Copy();
	return this;
}
Batch_Normalization* Batch_Normalization::Moving_Variance_Initializer(Initializer initializer) {
	if (moving_variance_initializer) {
		delete moving_variance_initializer;
	}
	moving_variance_initializer = initializer.Copy();
	return this;
}


Connection::Connection(Layer *layer, Layer *parent_layer, string properties) {
	unordered_map<int, int> weight_index;

	this->from_error = nullptr;
	this->from_neuron = nullptr;
	this->from_weight = nullptr;
	this->initializer = nullptr;
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->properties = properties;
	this->number_weights = 0;
	this->optimizer = nullptr;
	this->weight = nullptr;

	// set kernel / pool size if specified
	if (const char *kernel_size = strstr(properties.c_str(), "kernel")) {
		const char *end = strstr(kernel_size, ")");

		kernel_width = atoi(kernel_size + 7);
		kernel_size = strstr(kernel_size, "x");

		if (kernel_size && kernel_size < end && atoi(kernel_size + 1) > 0) {
			kernel_height = atoi(kernel_size + 1);
			kernel_size = strstr(kernel_size + 1, "x");

			if (kernel_size && kernel_size < end && atoi(kernel_size + 1) > 0) {
				kernel_depth = atoi(kernel_size + 1);
			}
			else {
				kernel_depth = 1;
			}
		}
		else {
			kernel_height = 1;
			kernel_depth = 1;
		}
	}
	else if (const char *pool_size = strstr(properties.c_str(), "pool")) {
		const char *end = strstr(pool_size, ")");

		kernel_width = atoi(pool_size + 5);
		pool_size = strstr(pool_size, "x");

		if (pool_size && pool_size < end && atoi(pool_size + 1) > 0) {
			kernel_height = atoi(pool_size + 1);
			pool_size = strstr(pool_size + 1, "x");

			if (pool_size && pool_size < end && atoi(pool_size + 1) > 0) {
				kernel_depth = atoi(pool_size + 1);
			}
			else {
				kernel_depth = 1;
			}
		}
		else {
			kernel_height = 1;
			kernel_depth = 1;
		}
	}
	else if (properties[0] == 'P') {
		kernel_width = (parent_layer->map_width > layer->map_width) ? (parent_layer->map_width / layer->map_width) : (layer->map_width / parent_layer->map_width);
		kernel_height = (parent_layer->map_height > layer->map_height) ? (parent_layer->map_height / layer->map_height) : (layer->map_height / parent_layer->map_height);
		kernel_depth = (parent_layer->map_depth > layer->map_depth) ? (parent_layer->map_depth / layer->map_depth) : (layer->map_depth / parent_layer->map_depth);
	}
	else {
		kernel_width = abs(parent_layer->map_width - layer->map_width) + 1;
		kernel_height = abs(parent_layer->map_height - layer->map_height) + 1;
		kernel_depth = abs(parent_layer->map_depth - layer->map_depth) + 1;
	}
	kernel_size = kernel_depth * kernel_height * kernel_width;

	// set stride size if specified
	if (const char *stride_size = strstr(properties.c_str(), "stride")) {
		const char *end = strstr(stride_size, ")");

		stride_width = atoi(stride_size + 7);
		stride_size = strstr(stride_size, "x");

		if (stride_size && stride_size < end && atoi(stride_size + 1) > 0) {
			stride_height = atoi(stride_size + 1);
			stride_size = strstr(stride_size + 1, "x");

			if (stride_size && stride_size < end && atoi(stride_size + 1) > 0) {
				stride_depth = atoi(stride_size + 1);
			}
			else {
				stride_depth = 1;
			}
		}
		else {
			stride_height = 1;
			stride_depth = 1;
		}
	}
	else if (properties[0] == 'P') {
		stride_width = (parent_layer->map_width > layer->map_width) ? (parent_layer->map_width / layer->map_width) : (layer->map_width / parent_layer->map_width);
		stride_height = (parent_layer->map_height > layer->map_height) ? (parent_layer->map_height / layer->map_height) : (layer->map_height / parent_layer->map_height);
		stride_depth = (parent_layer->map_depth > layer->map_depth) ? (parent_layer->map_depth / layer->map_depth) : (layer->map_depth / parent_layer->map_depth);
	}
	else {
		stride_width = 1;
		stride_height = 1;
		stride_depth = 1;
	}

	// allocate memory for the weight, if necessary
	if (properties[0] == 'W') {
		for (int j = 0, index = 0; j < layer->number_maps; j++) {
			for (int k = 0; k < parent_layer->number_maps; k++) {
				if (!strstr(properties.c_str(), "depthwise") || j % parent_layer->number_maps == k) {
					for (int l = 0; l < kernel_size; l++) {
						weight_index.insert(pair<int, int>(j * parent_layer->number_maps * kernel_size + k * kernel_size + l, index++));
					}
					number_weights += kernel_size;
				}
			}
		}
		cudaMalloc(&weight, sizeof(float) * number_weights);
		cudaMemset(weight, 0, sizeof(float) * number_weights);
	}

	if (properties[0] == 'P' || properties[0] == 'W') {
		int offset[3] = { kernel_depth - (abs(layer->map_depth * stride_depth - parent_layer->map_depth) + 1), kernel_height - (abs(layer->map_height * stride_height - parent_layer->map_height) + 1), kernel_width - (abs(layer->map_width * stride_width - parent_layer->map_width) + 1) };

		vector<Index> *from_error = new vector<Index>[parent_layer->number_nodes];
		vector<Index> *from_neuron = new vector<Index>[layer->number_nodes];
		vector<Index> *from_weight = (number_weights) ? (new vector<Index>[number_weights]) : (nullptr);

		for (int j = 0; j < layer->number_maps; j++) {
			for (int k = 0; k < layer->map_depth; k++) {
				for (int l = 0; l < layer->map_height; l++) {
					for (int m = 0; m < layer->map_width; m++) {
						int node_index[2] = { j * layer->map_size + k * layer->map_height * layer->map_width + l * layer->map_width + m, };

						if (properties[0] == 'W') {
							for (int n = 0; n < parent_layer->number_maps; n++) {
								if (!strstr(properties.c_str(), "depthwise") || j % parent_layer->number_maps == n) {
									int distance[3];

									for (int o = 0; o < parent_layer->map_depth; o++) {
										distance[0] = (layer->map_depth < parent_layer->map_depth) ? (o - k * stride_depth) : (k - o * stride_depth);
										if (-offset[0] <= distance[0] && distance[0] < kernel_depth - offset[0]) {
											for (int p = 0; p < parent_layer->map_height; p++) {
												distance[1] = (layer->map_height < parent_layer->map_height) ? (p - l * stride_height) : (l - p * stride_height);
												if (-offset[1] <= distance[1] && distance[1] < kernel_height - offset[1]) {
													for (int q = 0; q < parent_layer->map_width; q++) {
														distance[2] = (layer->map_width < parent_layer->map_width) ? (q - m * stride_width) : (m - q * stride_width);
														if (-offset[2] <= distance[2] && distance[2] < kernel_width - offset[2]) {
															Index index;

															node_index[1] = n * parent_layer->map_size + o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

															index.prev_node = node_index[1];
															index.next_node = node_index[0];
															index.weight = weight_index.find(j * parent_layer->number_maps * kernel_size + n * kernel_size + (distance[0] + offset[0]) * kernel_height * kernel_width + (distance[1] + offset[1]) * kernel_width + (distance[2] + offset[2]))->second;

															from_error[node_index[1]].push_back(index);
															from_neuron[node_index[0]].push_back(index);
															from_weight[index.weight].push_back(index);
														}
													}
												}
											}
										}
									}
								}
							}
						}
						else if (properties[0] == 'P') {
							int distance[3];

							for (int o = 0; o < parent_layer->map_depth; o++) {
								distance[0] = (layer->map_depth < parent_layer->map_depth) ? (o - k * stride_depth) : (k - o * stride_depth);
								if (0 <= distance[0] && distance[0] < kernel_depth) {
									for (int p = 0; p < parent_layer->map_height; p++) {
										distance[1] = (layer->map_height < parent_layer->map_height) ? (p - l * stride_height) : (l - p * stride_height);
										if (0 <= distance[1] && distance[1] < kernel_height) {
											for (int q = 0; q < parent_layer->map_width; q++) {
												distance[2] = (layer->map_width < parent_layer->map_width) ? (q - m * stride_width) : (m - q * stride_width);
												if (0 <= distance[2] && distance[2] < kernel_width) {
													Index index;

													node_index[1] = j * parent_layer->map_size + o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

													index.prev_node = node_index[1];
													index.next_node = node_index[0];
													index.weight = -1;

													from_error[node_index[1]].push_back(index);
													from_neuron[node_index[0]].push_back(index);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		{ // add last index (index.weight == -1)
			Index index;

			index.weight = -1;

			for (int j = 0; j < parent_layer->number_nodes; j++) {
				for (auto index = from_error[j].begin(); index != from_error[j].end(); index++) {
					if (index->weight == -1) {
						index->weight = from_error[j].size();
					}
				}
				from_error[j].push_back(index);
			}
			for (int j = 0; j < layer->number_nodes; j++) {
				for (auto index = from_neuron[j].begin(); index != from_neuron[j].end(); index++) {
					if (index->weight == -1) {
						index->weight = from_neuron[j].size();
					}
				}
				from_neuron[j].push_back(index);
			}
			if (from_weight) {
				for (int j = 0; j < this->number_weights; j++) {
					from_weight[j].push_back(index);
				}
			}
		}

		{ // copy from_error
			int max = 0;

			Index *memory = new Index[0];

			for (int j = 0; j < parent_layer->number_nodes; j++) {
				if (max < from_error[j].size()) {
					max = static_cast<int>(from_error[j].size());
				}
			}
			cudaMalloc(&this->from_error, sizeof(Index) * max * parent_layer->number_nodes);
			memory = new Index[max * parent_layer->number_nodes];

			for (int j = 0; j < parent_layer->number_nodes; j++) {
				for (int k = 0; k < from_error[j].size(); k++) {
					memory[j + k * parent_layer->number_nodes] = from_error[j][k];
				}
			}
			cudaMemcpy(this->from_error, memory, sizeof(Index) * max * parent_layer->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		{ // copry from_neuron
			int max = 0;

			Index *memory;

			for (int j = 0; j < layer->number_nodes; j++) {
				if (max < from_neuron[j].size()) {
					max = static_cast<int>(from_neuron[j].size());
				}
			}
			cudaMallocManaged(&this->from_neuron, sizeof(Index) * max * layer->number_nodes);
			memory = new Index[max * layer->number_nodes];

			for (int j = 0; j < layer->number_nodes; j++) {
				for (int k = 0; k < from_neuron[j].size(); k++) {
					memory[j + k * layer->number_nodes] = from_neuron[j][k];
				}
			}
			cudaMemcpy(this->from_neuron, memory, sizeof(Index) * max * layer->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		if (from_weight) {
			int max = 0;

			Index *memory;

			for (int j = 0; j < number_weights; j++) {
				if (max < from_weight[j].size()) {
					max = static_cast<int>(from_weight[j].size());
				}
			}
			cudaMallocManaged(&this->from_weight, sizeof(Index) * max * number_weights);
			memory = new Index[max * number_weights];

			for (int j = 0; j < number_weights; j++) {
				for (int k = 0; k < from_weight[j].size(); k++) {
					memory[j + k * number_weights] = from_weight[j][k];
				}
			}
			cudaMemcpy(this->from_weight, memory, sizeof(Index) * max * number_weights, cudaMemcpyHostToDevice);
			delete[] memory;
		}
		delete[] from_error;
		delete[] from_neuron;
		delete[] from_weight;
	}
}
Connection::~Connection() {}

void Connection::Destruct() {
	if (from_error) {
		cudaFree(from_error);
	}
	if (from_neuron) {
		cudaFree(from_neuron);
	}
	if (from_weight) {
		cudaFree(from_weight);
	}
	if (initializer) {
		delete initializer;
	}
	if (optimizer) {
		optimizer->Destruct();
		delete optimizer;
	}
	cudaFree(weight);
}
void Connection::Initialize() {
	if (initializer) {
		int kernel_size = this->kernel_size;
		int number_maps[] = { parent_layer->number_maps, layer->number_maps };
		int pool_size[] = { 1, 1 };

		Layer *layer = this->layer;
		Layer *parent_layer = this->parent_layer;

		if (strstr(properties.c_str(), "depthwise") && layer->Search_Child_Connection("pointwise")) {
			layer = layer->Search_Child_Connection("pointwise")->layer;
			number_maps[1] = layer->number_maps / parent_layer->number_maps;
			number_maps[0] = 1;
		}
		if (strstr(properties.c_str(), "pointwise") && parent_layer->Search_Connection("depthwise")) {
			kernel_size = parent_layer->Search_Connection("depthwise")->kernel_size;
			parent_layer = parent_layer->Search_Connection("depthwise")->parent_layer;

			number_maps[1] = layer->number_maps / parent_layer->number_maps;
			number_maps[0] = 1;
		}
		if (parent_layer->connection.size() > 0 && parent_layer->Search_Connection("P")) {
			pool_size[0] = parent_layer->Search_Connection("P")->kernel_size;
		}
		if (layer->child_connection.size() > 0 && layer->Search_Child_Connection("P")) {
			pool_size[1] = layer->Search_Child_Connection("P")->kernel_size;
		}
		initializer->Random(number_weights, weight, number_maps[0] * kernel_size * pool_size[0], number_maps[1] * kernel_size / pool_size[1]);
	}
}
void Connection::Optimizer(::Optimizer *optimizer) {
	if (this->optimizer) {
		this->optimizer->Destruct();
	}
	this->optimizer = optimizer->Copy(number_weights);
}

Connection *Connection::Initializer(::Initializer initializer) {
	if (this->initializer) {
		delete this->initializer;
	}
	this->initializer = initializer.Copy();
	return this;
}


Dropout::Dropout(int number_nodes, double rate) {
	this->batch_size = 1;
	cudaMalloc(&mask, number_nodes);
	this->number_nodes = number_nodes;
	this->rate = rate;
}
Dropout::~Dropout() {}

void Dropout::Destruct() {
	cudaFree(mask);
}
void Dropout::Initialize_Mask(int seed) {
	bool *memory = new bool[batch_size * number_nodes];

	default_random_engine *generator = ((seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));

	uniform_real_distribution<double> distribution(0, 1);

	for (int i = 0; i < batch_size * number_nodes; i++) {
		memory[i] = (rate == 0 || distribution(*generator) > rate) ? (true) : (false);
	}
	cudaMemcpy(mask, memory, batch_size * number_nodes, cudaMemcpyHostToDevice);
	delete[] memory;
}
void Dropout::Resize_Memory(int batch_size) {
	if (this->batch_size != batch_size) {
		cudaFree(mask);
		cudaMalloc(&mask, (this->batch_size = batch_size) * number_nodes);
	}
}


Initializer::Initializer(double value) {
	generator = nullptr;
	this->value = value;
	type = 0;
}
Initializer::Initializer(GlorotNormal initializer) {
	generator = initializer.generator;
	seed = initializer.seed;
	type = 4;
}
Initializer::Initializer(GlorotUniform initializer) {
	generator = initializer.generator;
	seed = initializer.seed;
	type = 3;
}
Initializer::Initializer(HeNormal initializer) {
	generator = initializer.generator;
	seed = initializer.seed;
	type = 6;
}
Initializer::Initializer(HeUniform initializer) {
	generator = initializer.generator;
	seed = initializer.seed;
	type = 5;
}
Initializer::Initializer(Orthogonal initializer) {
	generator = initializer.generator;
	seed = initializer.seed;
	value = initializer.gain;
	type = 7;
}
Initializer::Initializer(RandomNormal initializer) {
	generator = initializer.generator;
	mean = initializer.mean;
	stdv = initializer.stdv;
	seed = initializer.seed;
	type = 2;
}
Initializer::Initializer(RandomUniform initializer) {
	generator = initializer.generator;
	max = initializer.max;
	min = initializer.min;
	seed = initializer.seed;
	type = 1;
}
Initializer::~Initializer() {
	if (generator) {
		delete generator;
	}
}

void Initializer::Random(int memory_size, float _memory[], int fan_in, int fan_out) {
	float *memory = new float[memory_size];

	if (type == 0) { // Constant
		for (int i = 0; i < memory_size; i++) {
			memory[i] = value;
		}
	}
	else if (type == 1) { // RandomUniform
		uniform_real_distribution<double> distribution(min, max);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 2) { // RandomNormal
		normal_distribution<double> distribution(mean, stdv);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 3) { // GlorotUniform
		double range = sqrt(6.0 / (fan_in + fan_out));

		uniform_real_distribution<double> distribution(-range, range);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 4) { // GlorotNormal
		double stdv = sqrt(2.0 / (fan_in + fan_out));

		normal_distribution<double> distribution(0, stdv);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 5) { // HeUniform
		double range = sqrt(6.0 / fan_in);

		uniform_real_distribution<double> distribution(-range, range);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 6) { // HeNormal
		double stdv = sqrt(2.0 / fan_in);

		normal_distribution<double> distribution(0, stdv);

		for (int i = 0; i < memory_size; i++) {
			memory[i] = distribution(*generator);
		}
	}
	else if (type == 7) { // Orthogonal
		Matrix matrix(fan_in, fan_in);

		normal_distribution<double> distribution(0, 1);

		for (int i = 0; i < fan_in; i++) {
			for (int j = 0; j < fan_out; j++) {
				matrix(i, j) = distribution(*generator);
			}
		}
		matrix.Gram_Schmidt_Process(value);

		for (int i = 0, index = 0; i < fan_in; i++) {
			for (int j = 0; j < fan_out; j++) {
				memory[index++] = matrix(i, j);
			}
		}
	}

	cudaMemcpy(_memory, memory, sizeof(float) * memory_size, cudaMemcpyHostToDevice);
	delete[] memory;
}

Initializer* Initializer::Copy() {
	switch (type) {
	case 0: return new Initializer(value);
	case 1: return new Initializer(RandomUniform(min, max, seed));
	case 2: return new Initializer(RandomNormal(stdv, mean, seed));
	case 3: return new Initializer(GlorotUniform(seed));
	case 4: return new Initializer(GlorotNormal(seed));
	case 5: return new Initializer(HeUniform(seed));
	case 6: return new Initializer(HeNormal(seed));
	case 7: return new Initializer(Orthogonal(value, seed));
	}
	return nullptr;
}


Layer::Layer(int number_maps, string properties) {
	this->map_width = 1;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::Layer(int time_step, int number_maps, string properties) {
	this->map_width = 1;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, string properties) {
	this->map_width = map_width;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, int map_height, string properties) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::~Layer() {}

void Layer::Activate(bool training) {
	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	if (batch_normalization) {
		batch_normalization->Activate(neuron, training);
	}
	if (dropout) {
		if (training) {
			dropout->Initialize_Mask();
		}
		::Activate << <number_blocks, NUMBER_THREADS >> > (*this, *dropout, training);
	}
	else {
		::Activate << <number_blocks, NUMBER_THREADS >> > (*this);
	}
}
void Layer::Adjust_Parameter(int iterations) {
	if (batch_normalization) {
		batch_normalization->Adjust_Parameter(iterations);
	}

	// adjust bias
	if (bias) {
		::Adjust_Bias << <number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer, iterations);
	}

	// adjust weight
	for (int i = 0; i < connection.size(); i++) {
		Connection *connection = this->connection[i];

		if (connection->properties[0] == 'W') {
			::Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*connection->layer, *connection->parent_layer, *connection, *connection->optimizer, iterations);
		}
	}
}
void Layer::Backward() {
	for (int k = 0; k < connection.size(); k++) {
		Connection *connection = this->connection[k];

		dim3 number_blocks(connection->parent_layer->batch_size, connection->parent_layer->number_nodes / NUMBER_THREADS + 1);

		if (connection->properties[0] == 'P' && strstr(connection->properties.c_str(), "average")) {
			::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 0);
			continue;
		}
		if (connection->properties[0] == 'P' && strstr(connection->properties.c_str(), "max")) {
			::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 1);
			continue;
		}
		if (connection->properties[0] == 'W') {
			::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 2);
		}
	}
}
void Layer::Compile(::Optimizer *optimizer) {
	for (int j = 0; j < connection.size(); j++) {
		Connection *connection = this->connection[j];

		if (connection->properties[0] == 'W') {
			connection->Optimizer(optimizer);
			connection->Initialize();
		}
	}
	Optimizer(optimizer);
	Initialize();
}
void Layer::Construct() {
	this->activation = Activation::linear;
	this->batch_normalization = nullptr;
	this->batch_size = 1;
	this->initializer = nullptr;
	this->map_size = map_depth * map_height * map_width;
	this->number_nodes = number_maps * map_size;
	this->dropout = (strstr(properties.c_str(), "dropout")) ? (new Dropout(number_nodes, atof(strstr(properties.c_str(), "dropout") + 7))) : (nullptr);
	this->optimizer = nullptr;

	bias = nullptr;
	cudaMalloc(&error, sizeof(float) * number_nodes);
	cudaMalloc(&neuron, sizeof(float) * number_nodes);
}
void Layer::Differentiate(int loss, float **y_batch) {
	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	if (y_batch) {
		float *y_data, *memory = new float[batch_size * number_nodes];

		for (int h = 0; h < batch_size; h++) {
			memcpy(&memory[h * number_nodes], y_batch[h], sizeof(float) * number_nodes);
		}
		cudaMalloc(&y_data, sizeof(float) * batch_size * number_nodes);
		cudaMemcpy(y_data, memory, sizeof(float) * batch_size * number_nodes, cudaMemcpyHostToDevice);

		::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, loss, y_data);

		cudaFree(y_data);
		delete[] memory;

		Differentiate(loss);
	}
	else {
		if (dropout) {
			::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, *dropout);
		}
		else {
			::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this);
		}
		if (batch_normalization) {
			batch_normalization->Differentiate(error);
		}
	}
}
void Layer::Destruct() {
	if (batch_normalization) {
		batch_normalization->Destruct();
		delete batch_normalization;
	}
	if (bias) {
		cudaFree(bias);
	}
	if (dropout) {
		dropout->Destruct();
		delete dropout;
	}
	if (initializer) {
		delete initializer;
	}
	if (optimizer) {
		optimizer->Destruct();
		delete optimizer;
	}
	for (int i = 0; i < connection.size(); i++) {
		connection[i]->Destruct();
		delete connection[i];
	}
	cudaFree(error);
	cudaFree(neuron);
}
void Layer::Forward() {
	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	for (int k = 0; k < connection.size(); k++) {
		Connection *connection = this->connection[k];

		if (connection->properties[0] == 'P' && strstr(connection->properties.c_str(), "average")) {
			::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 0);
			continue;
		}
		if (connection->properties[0] == 'P' && strstr(connection->properties.c_str(), "max")) {
			::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 1);
			continue;
		}
		if (connection->properties[0] == 'W') {
			::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, 2);
		}
	}
}
void Layer::Initialize() {
	if (batch_normalization) {
		batch_normalization->Initialize();
	}
	if (bias) {
		initializer->Random(number_maps, bias, 1, number_maps);
	}
}
void Layer::Optimizer(::Optimizer *optimizer) {
	if (batch_normalization) {
		batch_normalization->Optimizer(*optimizer);
	}
	if (bias) {
		if (this->optimizer) {
			this->optimizer->Destruct();
			delete this->optimizer;
		}
		this->optimizer = optimizer->Copy(number_nodes);
	}
}
void Layer::Resize_Memory(int batch_size) {
	int memory_size = sizeof(float) * batch_size * number_nodes;

	if (this->batch_size != batch_size) {
		cudaFree(error);
		cudaFree(neuron);
		cudaMalloc(&error, memory_size);
		cudaMalloc(&neuron, memory_size);

		this->batch_size = batch_size;
	}
	if (batch_normalization) {
		batch_normalization->Resize_Memory(batch_size);
	}
	if (dropout) {
		dropout->Resize_Memory(batch_size);
	}
	cudaMemset(error, 0, memory_size);
	cudaMemset(neuron, 0, memory_size);
}

Batch_Normalization* Layer::Batch_Normalization(double epsilon, double momentum) {
	if (batch_normalization) {
		delete batch_normalization;
	}
	return (batch_normalization = new ::Batch_Normalization(number_maps, map_size, epsilon, momentum, this));
}

Connection* Layer::Search_Child_Connection(string properties) {
	for (int i = 0; i < child_connection.size(); i++) {
		if (strstr(child_connection[i]->properties.c_str(), properties.c_str())) {
			return child_connection[i];
		}
	}
	return nullptr;
}
Connection* Layer::Search_Connection(string properties) {
	for (int i = 0; i < connection.size(); i++) {
		if (strstr(connection[i]->properties.c_str(), properties.c_str())) {
			return connection[i];
		}
	}
	return nullptr;
}

Layer* Layer::Activation(int activation) {
	this->activation = activation;
	return this;
}
Layer* Layer::Copy() {
	return new Layer(number_maps, map_width, map_height, map_depth, properties);
}
Layer* Layer::Initializer(::Initializer initializer) {
	if (bias == nullptr) {
		cudaMalloc(&bias, sizeof(float) * number_nodes);
	}
	if (this->initializer) {
		delete this->initializer;
	}
	this->initializer = initializer.Copy();
	return this;
}


Matrix::Matrix(int number_rows, int number_columns) {
	this->number_columns = number_columns;
	this->number_rows = number_rows;

	memset(data = new double[number_rows * number_columns], 0, sizeof(double) * number_rows * number_columns);
}
Matrix::~Matrix() {
	delete[] data;
}

void Matrix::Gram_Schmidt_Process(double gain) {
	// Make Orthogonal Vectors
	for (int i = 0; i < number_columns; i++) {
		double *sum = { new double[i] };

		for (int j = 0; j < i; j++) {
			double temp[2] = { 0, };

			for (int k = 0; k < number_rows; k++) {
				temp[0] += (*this)(k, j) * (*this)(k, i);
				temp[1] += (*this)(k, j) * (*this)(k, j);
			}
			sum[j] = temp[0] / temp[1];
		}
		for (int j = 0; j < i; j++) {
			for (int k = 0; k < number_rows; k++) {
				(*this)(k, i) -= sum[j] * (*this)(k, j);
			}
		}
		delete[] sum;
	}

	// Make Orthonormal Vectors
	for (int i = 0; i < number_columns; i++) {
		double sum = 0;

		for (int j = 0; j < number_rows; j++) {
			sum += (*this)(j, i) * (*this)(j, i);
		}
		sum = sqrt(sum);
		for (int j = 0; j < number_rows; j++) {
			(*this)(j, i) = gain * (*this)(j, i) / sum;
		}
	}
}
void Matrix::Identity() {
	for (int i = 0; i < number_rows; i++) {
		for (int j = 0; j < number_columns; j++) {
			(*this)(i, j) = (i == j);
		}
	}
}
void Matrix::LQ_Decomposition(Matrix &L, Matrix &Q) {
	Matrix A = (*this);

	A.Transpose();
	A.QR_Decomposition(Q, L);
	Q.Transpose();
	L.Transpose();
}
void Matrix::QR_Decomposition(Matrix &Q, Matrix &R) {
	int m = number_rows;
	int n = number_columns;

	double *u = new double[m];
	double *v = new double[m];

	Matrix P(m, m);

	Q = Matrix(m, m);
	Q.Identity();
	R = (*this);

	for (int i = 0; i < n && i < m - 1; i++) {
		double sum[2] = { 0, };

		memset(u, 0, sizeof(double) * m);
		memset(v, 0, sizeof(double) * m);

		for (int j = i; j < m; j++) {
			u[j] = R(j, i);
			sum[0] += u[j] * u[j];
		}
		if (u[i]) {
			sum[0] = ((u[i] < 0) ? (-1) : (1)) * sqrt(sum[0]);
		}

		for (int j = i; j < m; j++) {
			v[j] = (j == i) ? (u[j] + sum[0]) : (u[j]);
			sum[1] += v[j] * v[j];
		}
		if (sum[1] = sqrt(sum[1])) {
			for (int j = i; j < m; j++) {
				v[j] /= sum[1];
			}
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < m; k++) {
					P(j, k) = ((j == k) ? (1) : (0)) - 2 * v[k] * v[j];
				}
			}
			R = P * R;
			Q = Q * P;
		}
	}
	delete[] u;
	delete[] v;
}
void Matrix::Transpose() {
	Matrix T(number_columns, number_rows);

	for (int i = 0; i < number_rows; i++) {
		for (int j = 0; j < number_columns; j++) {
			T(j, i) = (*this)(i, j);
		}
	}
	(*this) = T;
}

Matrix Matrix::Multiplication(const Matrix &A, const Matrix &B) {
	if (A.number_columns != B.number_rows) {
		cerr << "[Multiplication], A(" << A.number_rows << "x" << A.number_columns << ") * B(" << B.number_rows << "x" << B.number_columns << ") failed." << endl;
	}

	Matrix C(A.number_rows, B.number_columns);

	for (int i = 0; i < A.number_rows; i++) {
		for (int j = 0; j < B.number_columns; j++) {
			double sum = 0;

			for (int k = 0; k < A.number_columns; k++) {
				sum += A(i, k) * B(k, j);
			}
			C(i, j) = sum;
		}
	}
	return C;
}


Optimizer::Optimizer(int type, double decay, double epsilon, double learning_rate, double momentum_1, double momentum_2, int number_parameters) {
	Construct(type, decay, epsilon, learning_rate, momentum_1, momentum_2, number_parameters);
}
Optimizer::Optimizer(SGD SGD) {
	Construct(0, SGD.decay, 0, SGD.learning_rate, 0, 0, 0);
}
Optimizer::Optimizer(Momentum Momentum) {
	Construct(1, Momentum.decay, 0, Momentum.learning_rate, Momentum.momentum, 0, 0);
}
Optimizer::Optimizer(Nesterov Nesterov) {
	Construct(2, Nesterov.decay, 0, Nesterov.learning_rate, Nesterov.momentum, 0, 0);
}
Optimizer::Optimizer(Adam Adam) {
	Construct(3, Adam.decay, Adam.epsilon, Adam.learning_rate, Adam.momentum[0], Adam.momentum[1], 0);
}
Optimizer::~Optimizer() {}

void Optimizer::Construct(int type, double decay, double epsilon, double learning_rate, double momentum_1, double momentum_2, int number_parameters) {
	this->decay = decay;
	this->epsilon = epsilon;
	this->gradient = nullptr;
	this->memory[0] = nullptr;
	this->memory[1] = nullptr;
	this->learning_rate = learning_rate;
	this->momentum[0] = momentum_1;
	this->momentum[1] = momentum_2;
	this->type = type;

	if (type == 1 || type == 2) {
		cudaMalloc(&gradient, sizeof(float) * number_parameters);
		cudaMemset(gradient, 0, sizeof(float) * number_parameters);
	}
	else if (type == 3) {
		cudaMalloc(&memory[0], sizeof(float) * number_parameters);
		cudaMalloc(&memory[1], sizeof(float) * number_parameters);
		cudaMemset(memory[0], 0, sizeof(float) * number_parameters);
		cudaMemset(memory[1], 0, sizeof(float) * number_parameters);
	}
}
void Optimizer::Destruct() {
	if (gradient) {
		cudaFree(gradient);
		gradient = nullptr;
	}
	if (memory[0]) {
		cudaFree(memory[0]);
		memory[0] = nullptr;
	}
	if (memory[1]) {
		cudaFree(memory[1]);
		memory[1] = nullptr;
	}
}

double Optimizer::Calculate_Gradient(int index, double gradient, int iterations) {
	double learning_rate = this->learning_rate / (1 + decay * iterations);
	double output = 0;

	if (type == 0) { // SGD
		output = -learning_rate * gradient;
	}
	else if (type == 1) { // Momentum
		output = this->gradient[index] * momentum[0] - learning_rate * gradient;
		this->gradient[index] = output;
	}
	else if (type == 2) { // Nesterov
		output = (this->gradient[index] * momentum[0] - learning_rate * gradient) * momentum[0] - learning_rate * gradient;
		this->gradient[index] = output;
	}
	else if (type == 3) { // Adam
		double m1 = momentum[0] * memory[0][index] + (1 - momentum[0]) * gradient;
		double m2 = momentum[1] * memory[1][index] + (1 - momentum[1]) * gradient * gradient;

		memory[0][index] = m1;
		memory[1][index] = m2;
		m1 /= (1 - momentum[0]);
		m2 /= (1 - momentum[1]);
		output = -learning_rate * m1 / sqrt(m2 + epsilon);
	}
	return output;
}

Optimizer* Optimizer::Copy(int number_parameters) {
	return new Optimizer(type, decay, epsilon, learning_rate, momentum[0], momentum[1], number_parameters);
}


void Neural_Networks::Resize_Memory(int batch_size) {
	for (int i = 0; i < layer.size(); i++) {
		layer[i]->Resize_Memory(batch_size);
	}
	this->batch_size = batch_size;
}

double Neural_Networks::Calculate_Loss(Layer *layer, float **y_batch) {
	float loss, *y_data, *memory = new float[layer->batch_size * layer->number_nodes];

	cudaMalloc(&y_data, sizeof(float) * layer->batch_size * layer->number_nodes);

	for (int h = 0; h < batch_size; h++) {
		memcpy(&memory[h * layer->number_nodes], y_batch[h], sizeof(float) * layer->number_nodes);
	}
	cudaMemcpy(y_data, memory, sizeof(float) * layer->batch_size * layer->number_nodes, cudaMemcpyHostToDevice);
	::Calculate_Loss << <1, NUMBER_THREADS >> > (*layer, this->loss, y_data);
	cudaMemcpy(&loss, y_data, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(y_data);
	delete[] memory;

	return loss;
}

Neural_Networks::Neural_Networks() {
	batch_size = 1;
	optimizer = nullptr;
}
Neural_Networks::~Neural_Networks() {
	for (int i = 0; i < layer.size(); i++) {
		delete layer[i];
	}
}

void Neural_Networks::Compile(int loss, Optimizer optimizer) {
	this->loss = loss;

	if (this->optimizer) {
		this->optimizer->Destruct();
		delete this->optimizer;
	}
	this->iterations = 0;
	this->loss = loss;
	this->optimizer = optimizer.Copy();

	for (int i = 0; i < layer.size(); i++) {
		layer[i]->Compile(optimizer.Copy());
	}
}
void Neural_Networks::Predict(float input[], float output[]) {
	Predict(&input, &output);
}
void Neural_Networks::Predict(float **input, float **output, int batch_size) {
	float *memory = new float[batch_size * ((layer.front()->number_nodes > layer.back()->number_nodes) ? (layer.front()->number_nodes) : (layer.back()->number_nodes))];

	Resize_Memory(batch_size);

	for (int h = 0, i = 0; h < batch_size; h++) {
		memcpy(&memory[h * layer[i]->number_nodes], input[h], sizeof(float) * layer[i]->number_nodes);
	}
	cudaMemcpy(layer.front()->neuron, memory, sizeof(float) * batch_size * layer.front()->number_nodes, cudaMemcpyHostToDevice);

	for (int i = 1; i < layer.size(); i++) {
		layer[i]->Forward();
		layer[i]->Activate();
	}
	cudaMemcpy(memory, layer.back()->neuron, sizeof(float) * batch_size * layer.back()->number_nodes, cudaMemcpyDeviceToHost);

	for (int h = 0, i = layer.size() - 1; h < batch_size; h++) {
		memcpy(output[h], &memory[h * layer[i]->number_nodes], sizeof(float) * layer[i]->number_nodes);
	}
	delete[] memory;
}

double Neural_Networks::Evaluate(float **x_test, float **y_test, int test_size, int batch_size) {
	float **x_batch = new float*[batch_size];
	float **y_batch = new float*[batch_size];

	double loss = 0;

	for (int g = 0, h = 0; g < test_size; g++) {
		x_batch[h] = x_test[g];
		y_batch[h] = y_test[g];

		if (++h == batch_size || g == test_size - 1) {
			float *memory = new float[h * layer[0]->number_nodes];

			Resize_Memory(h);

			// copy x_test to neuron
			while (--h >= 0) {
				memcpy(&memory[h * layer[0]->number_nodes], x_batch[h], sizeof(float) * layer[0]->number_nodes);
			}
			cudaMemcpy(layer[0]->neuron, memory, sizeof(float) * this->batch_size * layer[0]->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;
			h = 0;

			// forward propagation
			for (int i = 1; i < layer.size(); i++) {
				layer[i]->Forward();
				layer[i]->Activate();
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch);
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / test_size;
}
double Neural_Networks::Fit(float **x_train, float **y_train, int train_size, int batch_size) {
	float **x_batch = new float*[batch_size];
	float **y_batch = new float*[batch_size];

	double loss = 0;

	for (int g = 0, h = 0; g < train_size; g++) {
		x_batch[h] = x_train[g];
		y_batch[h] = y_train[g];

		if (++h == batch_size || g == train_size - 1) {
			float *memory = new float[h * layer[0]->number_nodes];

			Resize_Memory(h);

			// copy x_train to neuron
			while (--h >= 0) {
				memcpy(&memory[h * layer[0]->number_nodes], x_batch[h], sizeof(float) * layer[0]->number_nodes);
			}
			cudaMemcpy(layer[0]->neuron, memory, sizeof(float) * this->batch_size * layer[0]->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;
			h = 0;

			// forward propagation
			for (int i = 1; i < layer.size(); i++) {
				layer[i]->Forward();
				layer[i]->Activate(true);
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch);

			// calculate error
			layer.back()->Differentiate(this->loss, y_batch);

			// error backpropagation
			layer.back()->Backward();

			for (int i = layer.size() - 2; i > 0; i--) {
				layer[i]->Differentiate();
				layer[i]->Backward();
			}

			// adjust parameter
			for (int i = 0; i < layer.size(); i++) {
				layer[i]->Adjust_Parameter(iterations);
			}
			iterations++;
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / train_size;
}

Connection* Neural_Networks::Connect(int from, int to, string properties) {
	Connection *connection = new Connection(layer[from], layer[to], properties);

	if (properties[0] == 'W') {
		if (!strstr(properties.c_str(), "depthwise") && layer[from]->bias == nullptr) {
			Layer *layer = this->layer[from];

			cudaMalloc(&layer->bias, sizeof(float) * layer->number_maps);
			layer->initializer = new Initializer(0);
		}
		connection->initializer = (strstr(properties.c_str(), "recurrent")) ? (new Initializer(Orthogonal())) : (new Initializer(GlorotUniform()));
	}
	if (properties[0] == 'P' && !(strstr(properties.c_str(), "average") || strstr(properties.c_str(), "max"))) {
		cerr << "[Connect], The pooling layer must have 'average' or 'max' property" << endl;
		return nullptr;
	}
	layer[from]->connection.push_back(connection);
	layer[to]->child_connection.push_back(connection);
	return connection;
}

Layer* Neural_Networks::Add(int number_nodes, string properties) {
	return Add(number_nodes, 1, 1, 1, properties);
}
Layer* Neural_Networks::Add(int number_maps, int map_width, string properties) {
	return Add(number_maps, map_width, 1, 1, properties);
}
Layer* Neural_Networks::Add(int number_maps, int map_width, int map_height, string properties) {
	return Add(number_maps, map_width, map_height, 1, properties);
}
Layer* Neural_Networks::Add(int number_maps, int map_width, int map_height, int map_depth, string properties) {
	Layer *layer = new Layer(1, number_maps, map_width, map_height, map_depth, properties);

	this->layer.push_back(layer);
	return layer;
}
Layer* Neural_Networks::Add(Layer layer) {
	this->layer.push_back(layer.Copy());
	return this->layer.back();
}