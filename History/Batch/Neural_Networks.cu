#include <math.h>
#include <memory.h>
#include <random>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "Neural_Networks.h"

#define NUMBER_THREADS 64

__global__ void Activate(Layer layer) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];

		neuron[j] = neuron[j] + layer.bias[j];
		neuron[j] = 1 / (1 + exp(-neuron[j]));
	}
}
__global__ void Adjust_Bias(Layer layer, double learning_rate) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		double sum = 0;

		for (int h = 0; h < layer.batch_size; h++) {
			sum += layer.error[h * layer.number_nodes + j];
		}
		layer.bias[j] -= learning_rate * sum;
	}
}
__global__ void Adjust_Weight(Layer layer, Layer parent_layer, Connection connection, double learning_rate) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < connection.number_weights) {
		double sum = 0;

		for (int k = j / parent_layer.number_nodes, l = j % parent_layer.number_nodes, h = 0; h < layer.batch_size; h++) {
			sum += layer.error[h * layer.number_nodes + k] * parent_layer.neuron[h * parent_layer.number_nodes + l];
		}
		connection.weight[j] -= learning_rate * sum;
	}
}
__global__ void Backward(Layer layer, Layer parent_layer, Connection connection) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < parent_layer.number_nodes) {
		float *error = &layer.error[blockIdx.x * layer.number_nodes];
		float *prev_error = &parent_layer.error[blockIdx.x * parent_layer.number_nodes];

		double sum = 0;

		for (int l = 0; l < layer.number_nodes; l++) {
			sum += error[l] * connection.weight[l * parent_layer.number_nodes + j];
		}
		prev_error[j] += sum;
	}
}
__global__ void Calculate_Error(Layer layer, float y_data[]) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		int index = blockIdx.x * layer.number_nodes + j;

		layer.error[index] = 2 * (layer.neuron[index] - y_data[index]) / (layer.batch_size * layer.number_nodes);
	}
}
__global__ void Calculate_Loss(Layer layer, float y_data[]) {
	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;

	for (int j = threadIdx.x; j < layer.batch_size * layer.number_nodes; j += blockDim.x) {
		sum[threadIdx.x] += (layer.neuron[j] - y_data[j]) * (layer.neuron[j] - y_data[j]);
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
__global__ void Differentiate(Layer layer) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		float *error = &layer.error[blockIdx.x * layer.number_nodes];
		float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];

		error[j] *= (1 - neuron[j]) * neuron[j];
	}
}
__global__ void Forward(Layer layer, Layer parent_layer, Connection connection) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		float *neuron = &layer.neuron[blockIdx.x * layer.number_nodes];
		float *prev_neuron = &parent_layer.neuron[blockIdx.x * parent_layer.number_nodes];

		double sum = 0;

		for (int l = 0; l < parent_layer.number_nodes; l++) {
			sum += prev_neuron[l] * connection.weight[j * parent_layer.number_nodes + l];
		}
		neuron[j] = sum;
	}
}

Connection::Connection(Layer *layer, Layer *parent_layer, double scale) {
	float *memory;

	this->layer = layer;
	this->parent_layer = parent_layer;
	this->number_weights = layer->number_nodes * parent_layer->number_nodes;

	memory = new float[number_weights];
	cudaMalloc(&weight, sizeof(float) * number_weights);

	for (int i = 0; i < number_weights; i++) {
		memory[i] = scale * (2.0 * rand() / RAND_MAX - 1);
	}
	cudaMemcpy(weight, memory, sizeof(float) * number_weights, cudaMemcpyHostToDevice);
	delete[] memory;
}
Connection::~Connection() {}

void Connection::Destruct() {
	cudaFree(weight);
}


Layer::Layer(int number_nodes) {
	this->batch_size = 1;
	this->number_nodes = number_nodes;

	cudaMalloc(&bias, sizeof(float) * number_nodes);
	cudaMemset(bias, 0, sizeof(float) * number_nodes);
	cudaMalloc(&error, sizeof(float) * number_nodes);
	cudaMalloc(&neuron, sizeof(float) * number_nodes);
}
Layer::~Layer() {}

void Layer::Destruct() {
	for (int i = 0; i < connection.size(); i++) {
		connection[i]->Destruct();
		delete connection[i];
	};
	cudaFree(bias);
	cudaFree(error);
	cudaFree(neuron);
}
void Layer::Forward() {
	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	for (int k = 0; k < connection.size(); k++) {
		::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *connection[k]->parent_layer, *connection[k]);
	}
	::Activate << <number_blocks, NUMBER_THREADS >> >(*this);
}
void Layer::Resize_Memory(int batch_size) {
	this->batch_size = batch_size;

	cudaFree(error);
	cudaFree(neuron);
	cudaMalloc(&error, sizeof(float) * batch_size * number_nodes);
	cudaMalloc(&neuron, sizeof(float) * batch_size * number_nodes);
}


void Neural_Networks::Resize_Memory(int batch_size) {
	if (this->batch_size != batch_size) {
		for (int i = 0; i < layer.size(); i++) {
			layer[i]->Resize_Memory(batch_size);
		}
		this->batch_size = batch_size;
	}
}

double Neural_Networks::Calculate_Loss(Layer *layer, float **y_batch) {
	float loss, *y_data, *memory = new float[layer->batch_size * layer->number_nodes];

	cudaMalloc(&y_data, sizeof(float) * layer->batch_size * layer->number_nodes);

	for (int h = 0; h < batch_size; h++) {
		memcpy(&memory[h * layer->number_nodes], y_batch[h], sizeof(float) * layer->number_nodes);
	}
	cudaMemcpy(y_data, memory, sizeof(float) * layer->batch_size * layer->number_nodes, cudaMemcpyHostToDevice);
	::Calculate_Loss << <1, NUMBER_THREADS >> > (*layer, y_data);
	cudaMemcpy(&loss, y_data, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(y_data);
	delete[] memory;

	return loss;
}

Neural_Networks::Neural_Networks() {
	batch_size = 1;
}
Neural_Networks::~Neural_Networks() {
	for (int i = 0; i < layer.size(); i++) {
		layer[i]->Destruct();
		delete layer[i];
	}
}

void Neural_Networks::Add(int number_nodes) {
	this->layer.push_back(new Layer(number_nodes));
}
void Neural_Networks::Compile(double learning_rate) {
	this->learning_rate = learning_rate;
}
void Neural_Networks::Connect(int from, int to, double scale) {
	Connection *connection = new Connection(layer[from], layer[to], scale);

	this->connection.push_back(connection);
	layer[from]->connection.push_back(connection);
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
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch);
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / (test_size * layer.back()->number_nodes);
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

			// initialize error to zero
			for (int i = 0; i < layer.size(); i++) {
				cudaMemset(layer[i]->error, 0, sizeof(float) * layer[i]->batch_size * layer[i]->number_nodes);
			}

			// forward propagation
			for (int i = 1; i < layer.size(); i++) {
				layer[i]->Forward();
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch);

			// error backpropagation
			for (int i = layer.size() - 1; i > 0; i--) {
				Layer *layer = this->layer[i];

				if (i == this->layer.size() - 1) {
					dim3 number_blocks(layer->batch_size, layer->number_nodes / NUMBER_THREADS + 1);

					float *y_data, *memory = new float[layer->batch_size * layer->number_nodes];

					cudaMalloc(&y_data, sizeof(float) * layer->batch_size * layer->number_nodes);

					for (int h = 0; h < layer->batch_size; h++) {
						memcpy(&memory[h * layer->number_nodes], y_batch[h], sizeof(float) * layer->number_nodes);
					}
					cudaMemcpy(y_data, memory, sizeof(float) * layer->batch_size * layer->number_nodes, cudaMemcpyHostToDevice);
					::Calculate_Error << <number_blocks, NUMBER_THREADS >> > (*layer, y_data);

					cudaFree(y_data);
					delete[] memory;
				}
				else {
					// backpropagate error
					for (int k = 0; k < layer->connection.size(); k++) {
						Connection *connection = layer->connection[k];

						dim3 number_blocks(connection->parent_layer->batch_size, connection->parent_layer->number_nodes / NUMBER_THREADS + 1);

						::Backward << <number_blocks, NUMBER_THREADS >> > (*layer, *connection->parent_layer, *connection);
					}
				}

				dim3 number_blocks(layer->batch_size, layer->number_nodes / NUMBER_THREADS + 1);

				::Differentiate << <number_blocks, NUMBER_THREADS >> > (*layer);
			}

			// adjust bias
			for (int i = 0; i < layer.size(); i++) {
				::Adjust_Bias << <layer[i]->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*layer[i], learning_rate);
			}

			// adjust weight
			for (int i = 0; i < connection.size(); i++) {
				Connection *connection = this->connection[i];

				::Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*connection->layer, *connection->parent_layer, *connection, learning_rate);
			}
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / (train_size * layer.back()->number_nodes);
}
