#include <math.h>
#include <memory.h>
#include <random>
#include <stdlib.h>

#include "Neural_Networks.h"

Connection::Connection(Layer *layer, Layer *parent_layer, double scale) {
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->number_weights = layer->number_nodes * parent_layer->number_nodes;

	weight = new float[number_weights];

	for (int i = 0; i < number_weights; i++) {
		weight[i] = scale * (2.0 * rand() / RAND_MAX - 1);
	}
}
Connection::~Connection() {
	delete[] weight;
}


Layer::Layer(int number_nodes) {
	this->number_nodes = number_nodes;

	memset(bias = new float[number_nodes], 0, sizeof(float) * number_nodes);
	error = new float[number_nodes];
	neuron = new float[number_nodes];
}
Layer::~Layer() {
	for (int i = 0; i < connection.size(); i++) {
		delete connection[i];
	}
	delete[] bias;
	delete[] error;
	delete[] neuron;
}

void Layer::Forward() {
	for (int j = 0; j < number_nodes; j++) {
		double sum = 0;

		for (int k = 0; k < connection.size(); k++) {
			Connection *connection = this->connection[k];

			Layer *parent_layer = connection->parent_layer;

			for (int l = 0; l < parent_layer->number_nodes; l++) {
				sum += parent_layer->neuron[l] * connection->weight[j * parent_layer->number_nodes + l];
			}
		}
		neuron[j] = sum + bias[j];

		// activate
		neuron[j] = 1 / (1 + exp(-neuron[j]));
	}
}


Neural_Networks::Neural_Networks() {}
Neural_Networks::~Neural_Networks() {
	for (int i = 0; i < layer.size(); i++) {
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
	memcpy(layer[0]->neuron, input, sizeof(float) * layer[0]->number_nodes);
	
	for (int i = 1; i < layer.size(); i++) {
		layer[i]->Forward();
	}
	memcpy(output, layer.back()->neuron, sizeof(float) * layer.back()->number_nodes);
}

double Neural_Networks::Evaluate(float **x_test, float **y_test, int test_size) {
	double loss = 0;

	for (int h = 0; h < test_size; h++) {
		memcpy(layer[0]->neuron, x_test[h], sizeof(float) * layer[0]->number_nodes);

		// forward propagation
		for (int i = 1; i < layer.size(); i++) {
			layer[i]->Forward();
		}

		// calculate loss
		for (int j = 0; j < layer.back()->number_nodes; j++) {
			loss += (layer.back()->neuron[j] - y_test[h][j]) * (layer.back()->neuron[j] - y_test[h][j]);
		}
	}
	return loss / (test_size * layer.back()->number_nodes);
}
double Neural_Networks::Fit(float **x_train, float **y_train, int train_size) {
	double loss = 0;

	for (int h = 0; h < train_size; h++) {
		// copy x_train to neuron
		memcpy(layer[0]->neuron, x_train[h], sizeof(float) * layer[0]->number_nodes);
		
		// initialize error to zero
		for (int i = 0; i < layer.size(); i++) {
			memset(layer[i]->error, 0, sizeof(float) * layer[i]->number_nodes);
		}

		// forward propagation
		for (int i = 1; i < layer.size(); i++) {
			layer[i]->Forward();
		}

		// error backpropagation
		for (int i = layer.size() - 1; i > 0; i--) {
			Layer *layer = this->layer[i];

			for (int j = 0; j < layer->number_nodes; j++) {
				if (i == this->layer.size() - 1) {
					// calculate error
					layer->error[j] = (layer->neuron[j] - y_train[h][j]);
					loss += layer->error[j] * layer->error[j];
					layer->error[j] = 2 * layer->error[j] / layer->number_nodes;
				}
				else {
					// backpropagate error
					for (int k = 0; k < layer->connection.size(); k++) {
						Connection *connection = layer->connection[k];

						Layer *parent_layer = connection->parent_layer;

						for (int l = 0; l < parent_layer->number_nodes; l++) {
							parent_layer->error[l] += layer->error[j] * connection->weight[j * parent_layer->number_nodes + l];
						}
					}
				}

				// differentiate
				layer->error[j] *= (1 - layer->neuron[j]) * layer->neuron[j];
			}
		}

		// adjust bias
		for (int i = 0; i < layer.size(); i++) {
			Layer *layer = this->layer[i];

			for (int j = 0; j < layer->number_nodes; j++) {
				layer->bias[j] -= learning_rate * layer->error[j];
			}
		}

		// adjust weight
		for (int i = 0; i < connection.size(); i++) {
			Connection *connection = this->connection[i];

			Layer *parent_layer = connection->parent_layer;

			for (int j = 0; j < connection->number_weights; j++) {
				connection->weight[j] -= learning_rate * connection->layer->error[j / parent_layer->number_nodes] * parent_layer->neuron[j % parent_layer->number_nodes];
			}
		}
	}
	return loss / (train_size * layer.back()->number_nodes);
}
