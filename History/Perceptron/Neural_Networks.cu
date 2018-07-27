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
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		layer.neuron[j] = layer.neuron[j] + layer.bias[j];
		layer.neuron[j] = 1 / (1 + exp(-layer.neuron[j]));
	}
}
__global__ void Adjust_Bias(Layer layer, double learning_rate) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		layer.bias[j] -= learning_rate * layer.error[j];
	}
}
__global__ void Adjust_Weight(Layer layer, Layer parent_layer, Connection connection, double learning_rate) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < connection.number_weights) {
		connection.weight[j] -= learning_rate * layer.error[j / parent_layer.number_nodes] * parent_layer.neuron[j % parent_layer.number_nodes];
	}
}
__global__ void Backward(Layer layer, Layer parent_layer, Connection connection) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < parent_layer.number_nodes) {
		double sum = 0;

		for (int l = 0; l < layer.number_nodes; l++) {
			sum += layer.error[l] * connection.weight[l * parent_layer.number_nodes + j];
		}
		parent_layer.error[j] += sum;
	}
}
__global__ void Calculate_Error(Layer layer, float y_data[]) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		layer.error[j] = 2 * (layer.neuron[j] - y_data[j]) / layer.number_nodes;
	}
}
__global__ void Calculate_Loss(Layer layer, float y_data[]) {
	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;

	for (int j = threadIdx.x; j < layer.number_nodes; j += blockDim.x) {
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
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		layer.error[j] *= (1 - layer.neuron[j]) * layer.neuron[j];
	}
}
__global__ void Forward(Layer layer, Layer parent_layer, Connection connection) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		double sum = 0;

		for (int l = 0; l < parent_layer.number_nodes; l++) {
			sum += parent_layer.neuron[l] * connection.weight[j * parent_layer.number_nodes + l];
		}
		layer.neuron[j] += sum;
	}
}

Connection::Connection(Layer *layer, Layer *parent_layer, double scale) {
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->number_weights = layer->number_nodes * parent_layer->number_nodes;

	cudaMalloc(&weight, sizeof(float) * number_weights);

	for (int i = 0; i < number_weights; i++) {
		float value = scale * (2.0 * rand() / RAND_MAX - 1);

		cudaMemcpy(&weight[i], &value, sizeof(float), cudaMemcpyHostToDevice);
	}
}
Connection::~Connection() {}

void Connection::Destruct() {
	cudaFree(weight);
}


Layer::Layer(int number_nodes) {
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
	for (int k = 0; k < connection.size(); k++) {
		::Forward << <number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *connection[k]->parent_layer, *connection[k]);
	}
	::Activate << <number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> >(*this);
}


double Neural_Networks::Calculate_Loss(Layer *layer, float _y_data[]) {
	float loss, *y_data;

	cudaMalloc(&y_data, sizeof(float) * layer->number_nodes);
	cudaMemcpy(y_data, _y_data, sizeof(float) * layer->number_nodes, cudaMemcpyHostToDevice);

	::Calculate_Loss << <1, NUMBER_THREADS >> > (*layer, y_data);

	cudaMemcpy(&loss, y_data, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(y_data);

	return loss;
}

Neural_Networks::Neural_Networks() {}
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
	cudaMemcpy(layer[0]->neuron, input, sizeof(float) * layer[0]->number_nodes, cudaMemcpyHostToDevice);

	for (int i = 1; i < layer.size(); i++) {
		layer[i]->Forward();
	}
	cudaMemcpy(output, layer.back()->neuron, sizeof(float) * layer.back()->number_nodes, cudaMemcpyDeviceToHost);
}

double Neural_Networks::Evaluate(float **x_test, float **y_test, int test_size) {
	double loss = 0;

	for (int h = 0; h < test_size; h++) {
		cudaMemcpy(layer[0]->neuron, x_test[h], sizeof(float) * layer[0]->number_nodes, cudaMemcpyHostToDevice);

		// forward propagation
		for (int i = 1; i < layer.size(); i++) {
			layer[i]->Forward();
		}
		loss += Calculate_Loss(layer.back(), y_test[h]);
	}
	return loss / (test_size * layer.back()->number_nodes);
}
double Neural_Networks::Fit(float **x_train, float **y_train, int train_size) {
	double loss = 0;

	for (int h = 0; h < train_size; h++) {
		// copy x_train to neuron
		cudaMemcpy(layer[0]->neuron, x_train[h], sizeof(float) * layer[0]->number_nodes, cudaMemcpyHostToDevice);

		// initialize error to zero
		for (int i = 0; i < layer.size(); i++) {
			cudaMemset(layer[i]->error, 0, sizeof(float) * layer[i]->number_nodes);
		}

		// forward propagation
		for (int i = 1; i < layer.size(); i++) {
			layer[i]->Forward();
		}
		loss += Calculate_Loss(layer.back(), y_train[h]);

		// error backpropagation
		for (int i = layer.size() - 1; i > 0; i--) {
			Layer *layer = this->layer[i];

			for (int j = 0; j < layer->number_nodes; j++) {
				if (i == this->layer.size() - 1) {
					float *y_data;

					cudaMalloc(&y_data, sizeof(float) * layer->number_nodes);
					cudaMemcpy(y_data, y_train[h], sizeof(float) * layer->number_nodes, cudaMemcpyHostToDevice);
					::Calculate_Error << <layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*layer, y_data);

					cudaFree(y_data);
				}
				else {
					// backpropagate error
					for (int k = 0; k < layer->connection.size(); k++) {
						Connection *connection = layer->connection[k];

						::Backward << <connection->parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*layer, *connection->parent_layer, *connection);
					}
				}
				::Differentiate << <layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*layer);
			}
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
	return loss / (train_size * layer.back()->number_nodes);
}
