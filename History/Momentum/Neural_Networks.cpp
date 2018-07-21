#include <math.h>
#include <memory.h>
#include <random>
#include <stdlib.h>

#include "Neural_Networks.h"

Connection::Connection(Layer *layer, Layer *parent_layer, double scale) {
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->number_weights = layer->number_nodes * parent_layer->number_nodes;
	this->optimizer = nullptr;

	weight = new float[number_weights];

	for (int i = 0; i < number_weights; i++) {
		weight[i] = scale * (2.0 * rand() / RAND_MAX - 1);
	}
}
Connection::~Connection() {
	if (optimizer) {
		delete optimizer;
	}
	delete[] weight;
}


Layer::Layer(int number_nodes, int activation, string properties) {
	this->activation = activation;
	this->batch_size = 1;
	this->number_nodes = number_nodes;
	this->optimizer = nullptr;
	this->properties = properties;

	memset(bias = new float[number_nodes], 0, sizeof(float) * number_nodes);
	error = new float[number_nodes];
	neuron = new float[number_nodes];
}
Layer::~Layer() {
	if (optimizer) {
		delete optimizer;
	}
	for (int i = 0; i < connection.size(); i++) {
		delete connection[i];
	}
	delete[] bias;
	delete[] error;
	delete[] neuron;
}

void Layer::Activation(bool training) {
	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		float *neuron = &this->neuron[h * number_nodes];

		for (int j = 0; j < number_nodes; j++) {
			neuron[j] += bias[j];
		}

		if (activation == Activation::softmax) {
			double max;
			double sum = 0;

			for (int j = 0; j < number_nodes; j++) {
				if (j == 0 || max < neuron[j]) {
					max = neuron[j];
				}
			}
			for (int j = 0; j < number_nodes; j++) {
				sum += (neuron[j] = exp(neuron[j] - max));
			}
			for (int j = 0; j < number_nodes; j++) {
				neuron[j] /= sum;
			}
		}
		else {
			if (activation == Activation::linear) {
				// neuron = neuron;
			}
			else if (activation == Activation::relu) {
				for (int j = 0; j < number_nodes; j++) {
					neuron[j] *= (neuron[j] > 0);
				}
			}
			else if (activation == Activation::sigmoid) {
				for (int j = 0; j < number_nodes; j++) {
					neuron[j] = 1 / (1 + exp(-neuron[j]));
				}
			}
			else if (activation == Activation::tanh) {
				for (int j = 0; j < number_nodes; j++) {
					neuron[j] = 2 / (1 + exp(-2 * neuron[j])) - 1;
				}
			}

			if (strstr(properties.c_str(), "dropout")) {
				double rate = atof(strstr(properties.c_str(), "dropout") + 7);

				for (int j = 0; j < number_nodes; j++) {
					neuron[j] *= (training) ? (mask[h * number_nodes + j]) : (1 - rate);
				}
			}
		}
	}
}
void Layer::Backward() {
	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		float *error = &this->error[h * number_nodes];
		float *neuron = &this->neuron[h * number_nodes];

		for (int j = 0; j < number_nodes; j++) {
			for (int k = 0; k < connection.size(); k++) {
				Connection *connection = this->connection[k];

				Layer *parent_layer = connection->parent_layer;

				float *next_error = &parent_layer->error[h * parent_layer->number_nodes];
				float *weight = &connection->weight[j * parent_layer->number_nodes];

				for (int l = 0; l < parent_layer->number_nodes; l++) {
					next_error[l] += error[j] * weight[l];
				}
			}
		}
	}
}
void Layer::Derivative(int loss, float **y_batch) {
	if (y_batch) {
		#pragma omp parallel for
		for (int h = 0; h < batch_size; h++) {
			float *error = &this->error[h * number_nodes];
			float *neuron = &this->neuron[h * number_nodes];

			if (loss == Loss::cross_entropy) {
				if (activation == Activation::sigmoid) {
					for (int j = 0; j < number_nodes; j++) {
						error[j] = (neuron[j] - y_batch[h][j]) / (number_nodes * batch_size);
					}
				}
				else if (activation == Activation::softmax) {
					for (int j = 0; j < number_nodes; j++) {
						error[j] = (neuron[j] - y_batch[h][j]) / batch_size;
					}
				}
			}
			else if (loss == Loss::mean_squared_error) {
				for (int j = 0; j < number_nodes; j++) {
					error[j] = 2 * (neuron[j] - y_batch[h][j]) / (number_nodes * batch_size);
				}
			}
		}
		Derivative(loss);
	}
	else {
		#pragma omp parallel for
		for (int h = 0; h < batch_size; h++) {
			float *error = &this->error[h * number_nodes];
			float *neuron = &this->neuron[h * number_nodes];

			if (activation == Activation::linear) {
				// error *= 1;
			}
			else if (activation == Activation::relu) {
				for (int j = 0; j < number_nodes; j++) {
					error[j] *= (neuron[j] > 0);
				}
			}
			else if (activation == Activation::sigmoid && loss != Loss::cross_entropy){
				for (int j = 0; j < number_nodes; j++) {
					error[j] *= (1 - neuron[j]) * neuron[j];
				}
			}
			else if (activation == Activation::tanh) {
				for (int j = 0; j < number_nodes; j++) {
					error[j] *= (1 - neuron[j]) * (1 + neuron[j]);
				}
			}

			if (strstr(properties.c_str(), "dropout")) {
				for (int j = 0; j < number_nodes; j++) {
					error[j] *= mask[h * number_nodes + j];
				}
			}
		}
	}
}
void Layer::Forward() {
	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		float *neuron = &this->neuron[h * number_nodes];

		for (int j = 0; j < number_nodes; j++) {
			double sum = 0;

			for (int k = 0; k < connection.size(); k++) {
				Connection *connection = this->connection[k];

				Layer *parent_layer = connection->parent_layer;

				float *prev_neuron = &parent_layer->neuron[h * parent_layer->number_nodes];
				float *weight = &connection->weight[j * parent_layer->number_nodes];

				for (int l = 0; l < parent_layer->number_nodes; l++) {
					sum += prev_neuron[l] * weight[l];
				}
			}
			neuron[j] = sum;
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
	double sum = 0, *batch_sum = new double[batch_size];

	memset(batch_sum, 0, sizeof(double) * batch_size);

	#pragma omp parallel for
	for (int h = 0; h < batch_size; h++) {
		float *neuron = &layer->neuron[h * layer->number_nodes];

		if (loss == Loss::cross_entropy) {
			if (layer->activation == Activation::sigmoid) {
				for (int j = 0; j < layer->number_nodes; j++) {
					batch_sum[h] -= (y_batch[h][j] * log(neuron[j] + 0.00000001) + (1 - y_batch[h][j]) * log(1.00000001 - neuron[j])) / layer->number_nodes;
				}
			}
			else if (layer->activation == Activation::softmax) {
				for (int j = 0; j < layer->number_nodes; j++) {
					batch_sum[h] -= y_batch[h][j] * log(neuron[j] + 0.00000001);
				}
			}
		}
		else if (loss == Loss::mean_squared_error) {
			for (int j = 0; j < layer->number_nodes; j++) {
				batch_sum[h] += (neuron[j] - y_batch[h][j]) * (neuron[j] - y_batch[h][j]) / layer->number_nodes;
			}
		}
	}
	for (int h = 0; h < batch_size; h++) {
		sum += batch_sum[h];
	}
	return sum;
}

Neural_Networks::Neural_Networks() {
	batch_size = 1;
	optimizer = nullptr;
}
Neural_Networks::~Neural_Networks() {
	if (optimizer) {
		delete optimizer;
	}
	for (int i = 0; i < layer.size(); i++) {
		delete layer[i];
	}
}

void Neural_Networks::Add(int number_nodes, int activation, string properties) {
	this->layer.push_back(new Layer(number_nodes, activation, properties));
}
void Neural_Networks::Compile(int loss, Optimizer *optimizer) {
	if (this->optimizer) {
		delete this->optimizer;
	}
	this->loss = loss;
	this->optimizer = optimizer;

	for (int i = 0; i < connection.size(); i++) {
		if (connection[i]->optimizer) {
			delete connection[i]->optimizer;
		}
		connection[i]->optimizer = optimizer->Copy(connection[i]->number_weights);
	}
	for (int i = 1; i < layer.size(); i++) {
		if (layer[i]->optimizer) {
			delete layer[i]->optimizer;
		}
		layer[i]->optimizer = optimizer->Copy(layer[i]->number_nodes);
	}
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
		layer[i]->Activation();
	}
	for (int h = 0, i = layer.size() - 1; h < batch_size; h++) {
		memcpy(output[h], &layer[i]->neuron[h * layer[i]->number_nodes], sizeof(float) * layer[i]->number_nodes);
	}
}

float** Neural_Networks::Shuffle(int seed, float **data, int data_size) {
	srand(seed);

	for (int i = 0, index; i < data_size; i++) {
		float *d = data[index = rand() % data_size];

		data[index] = data[i];
		data[i] = d;
	}
	return data;
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
				layer[i]->Activation();
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
			Resize_Memory(h);

			// prepare dropout_mask if dropout specified
			for (int i = 1; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				if (strstr(layer->properties.c_str(), "dropout")) {
					double rate = atof(strstr(layer->properties.c_str(), "dropout") + 7);

					layer->mask = new bool[h * layer->number_nodes];

					for (int j = 0; j < h * layer->number_nodes; j++) {
						layer->mask[j] = (rand() >= rate * (RAND_MAX + 1));
					}
				}
			}

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
				layer[i]->Activation(true);
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch);

			// calculate error
			layer.back()->Derivative(this->loss, y_batch);

			// error backpropagation
			layer.back()->Backward();

			for (int i = layer.size() - 2; i > 0; i--) {
				layer[i]->Derivative();
				layer[i]->Backward();
			}

			// adjust bias
			for (int i = 1; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				#pragma omp parallel for
				for (int j = 0; j < layer->number_nodes; j++) {
					double sum = 0;

					for (int h = 0; h < this->batch_size; h++) {
						sum += layer->error[h * layer->number_nodes + j];
					}
					layer->bias[j] += layer->optimizer->Calculate_Gradient(j, sum);
				}
			}

			// adjust weight
			for (int i = 0; i < connection.size(); i++) {
				Connection *connection = this->connection[i];

				Layer *layer = connection->layer;
				Layer *parent_layer = connection->parent_layer;

				#pragma omp parallel for
				for (int j = 0; j < connection->number_weights; j++) {
					int k = j / parent_layer->number_nodes;
					int l = j % parent_layer->number_nodes;

					double sum = 0;

					for (int h = 0; h < this->batch_size; h++) {
						sum += layer->error[h * layer->number_nodes + k] * parent_layer->neuron[h * parent_layer->number_nodes + l];
					}
					connection->weight[j] += connection->optimizer->Calculate_Gradient(j, sum);
				}
			}

			for (int i = 1; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				if (strstr(layer->properties.c_str(), "dropout")) {
					delete[] layer->mask;
				}
			}
		}
	}
	delete[] x_batch;
	delete[] y_batch;

	return loss / train_size;
}


Optimizer::Optimizer(int type, double learning_rate, double momentum, int number_parameters) {
	this->gradient = nullptr;
	this->learning_rate = learning_rate;
	this->momentum = momentum;
	this->type = type;

	if (type) {
		memset(gradient = new float[number_parameters], 0, sizeof(float) * number_parameters);
	}
}
Optimizer::~Optimizer() {
	delete[] gradient;
}

double Optimizer::Calculate_Gradient(int index, double gradient) {
	float &gradient = this->gradient[index];

	double sum = this->gradient[index] * momentum - learning_rate * gradient;

	gradient = sum;
	return sum;
}

Optimizer* Optimizer::Copy(int number_parameters) {
	return new Optimizer(type, learning_rate, momentum, number_parameters);
}
