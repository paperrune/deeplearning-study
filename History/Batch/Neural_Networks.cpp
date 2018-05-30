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
	this->batch_size = 1;
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
	for (int h = 0; h < batch_size; h++) {
		float *neuron = &this->neuron[h * number_nodes];

		for (int j = 0; j < number_nodes; j++) {
			double sum = 0;

			for (int k = 0; k < connection.size(); k++) {
				Connection *connection = this->connection[k];

				Layer *parent_layer = connection->parent_layer;

				float *prev_neuron = &parent_layer->neuron[h * parent_layer->number_nodes];

				for (int l = 0; l < parent_layer->number_nodes; l++) {
					sum += prev_neuron[l] * connection->weight[j * parent_layer->number_nodes + l];
				}
			}
			neuron[j] = sum + bias[j];

			// activate
			neuron[j] = 1 / (1 + exp(-neuron[j]));
		}
	}
}
void Layer::Resize_Memory(int batch_size) {
	this->batch_size = batch_size;

	error = (float*)realloc(error, sizeof(float) * batch_size * number_nodes);
	neuron = (float*)realloc(neuron, sizeof(float) * batch_size * number_nodes);
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
	double loss = 0;

	for (int h = 0; h < batch_size; h++) {
		float *neuron = &layer->neuron[h * layer->number_nodes];

		for (int j = 0; j < layer->number_nodes; j++) {
			loss += (neuron[j] - y_batch[h][j]) * (neuron[j] - y_batch[h][j]);
		}
	}
	return loss;
}

Neural_Networks::Neural_Networks() {
	batch_size = 1;
}
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
	Predict(&input, &output);
}
void Neural_Networks::Predict(float **input, float **output, int batch_size) {
	Resize_Memory(batch_size);

	for (int h = 0, i = 0; h < batch_size; h++) {
		memcpy(&layer[i]->neuron[h * layer[i]->number_nodes], input[h], sizeof(float) * layer[i]->number_nodes);
	}
	for (int i = 1; i < layer.size(); i++) {
		layer[i]->Forward();
	}
	for (int h = 0, i = layer.size() - 1; h < batch_size; h++) {
		memcpy(output[h], &layer[i]->neuron[h * layer[i]->number_nodes], sizeof(float) * layer[i]->number_nodes);
	}
}

double Neural_Networks::Evaluate(float **x_test, float **y_test, int test_size, int batch_size) {
	float **x_batch = new float*[batch_size];
	float **y_batch = new float*[batch_size];
	
	double loss = 0;

	for (int g = 0, h = 0; g < test_size; g++) {
		x_batch[h] = x_test[g];
		y_batch[h] = y_test[g];

		if (++h == batch_size || g == test_size - 1) {
			Resize_Memory(h);

			// copy x_test to neuron
			while (--h >= 0) {
				memcpy(&layer[0]->neuron[h * layer[0]->number_nodes], x_batch[h], sizeof(float) * layer[0]->number_nodes);
			}
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
			Resize_Memory(h);

			// copy x_train to neuron
			while (--h >= 0) {
				memcpy(&layer[0]->neuron[h * layer[0]->number_nodes], x_batch[h], sizeof(float) * layer[0]->number_nodes);
			}
			h = 0;

			// initialize error to zero
			for (int i = 0; i < layer.size(); i++) {
				memset(layer[i]->error, 0, sizeof(float) * this->batch_size * layer[i]->number_nodes);
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

				for (int h = 0; h < this->batch_size; h++) {
					float *error = &layer->error[h * layer->number_nodes];
					float *neuron = &layer->neuron[h * layer->number_nodes];

					for (int j = 0; j < layer->number_nodes; j++) {
						if (i == this->layer.size() - 1) {
							// calculate error
							error[j] = 2 * (neuron[j] - y_batch[h][j]) / (layer->number_nodes * this->batch_size);
						}
						else {
							// backpropagate error
							for (int k = 0; k < layer->connection.size(); k++) {
								Connection *connection = layer->connection[k];

								Layer *parent_layer = connection->parent_layer;

								float *next_error = &parent_layer->error[h * parent_layer->number_nodes];

								for (int l = 0; l < parent_layer->number_nodes; l++) {
									next_error[l] += error[j] * connection->weight[j * parent_layer->number_nodes + l];
								}
							}
						}

						// differentiate
						error[j] *= (1 - neuron[j]) * neuron[j];
					}
				}
			}

			// adjust bias
			for (int i = 0; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				for (int h = 0; h < this->batch_size; h++) {
					float *error = &layer->error[h * layer->number_nodes];

					for (int j = 0; j < layer->number_nodes; j++) {
						layer->bias[j] -= learning_rate * error[j];
					}
				}
			}

			// adjust weight
			for (int i = 0; i < connection.size(); i++) {
				Connection *connection = this->connection[i];

				Layer *layer = connection->layer;
				Layer *parent_layer = connection->parent_layer;

				for (int h = 0; h < this->batch_size; h++) {
					float *error = &layer->error[h * layer->number_nodes];
					float *neuron = &parent_layer->neuron[h * parent_layer->number_nodes];

					for (int j = 0; j < connection->number_weights; j++) {
						connection->weight[j] -= learning_rate * error[j / parent_layer->number_nodes] * neuron[j % parent_layer->number_nodes];
					}
				}
			}
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / (train_size * layer.back()->number_nodes);
}