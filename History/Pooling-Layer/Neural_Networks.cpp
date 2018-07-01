#include <math.h>
#include <memory.h>
#include <random>
#include <stdlib.h>
#include <unordered_map>

#include "Neural_Networks.h"

Connection::Connection(Layer *layer, Layer *parent_layer, string properties) {
	unordered_map<int, int> weight_index;

	this->from_error = nullptr;
	this->from_neuron = nullptr;
	this->from_weight = nullptr;
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->properties = properties;
	this->number_weights = 0;
	this->optimizer = nullptr;
	this->weight = nullptr;

	// set kernel size if specified
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
				for (int l = 0; l < kernel_size; l++) {
					weight_index.insert(pair<int, int>(j * parent_layer->number_maps * kernel_size + k * kernel_size + l, index++));
				}
				number_weights += kernel_size;
			}
		}
		memset(weight = new float[number_weights], 0, sizeof(float) * number_weights);
	}

	if (properties[0] == 'P' || properties[0] == 'W') {
		from_error = new vector<Index>[parent_layer->number_nodes];
		from_neuron = new vector<Index>[layer->number_nodes];
		from_weight = new vector<Index>[number_weights];

		for (int j = 0; j < layer->number_maps; j++) {
			for (int k = 0; k < layer->map_depth; k++) {
				for (int l = 0; l < layer->map_height; l++) {
					for (int m = 0; m < layer->map_width; m++) {
						int node_index[2] = { j * layer->map_size + k * layer->map_height * layer->map_width + l * layer->map_width + m, };

						if (properties[0] == 'W') {
							for (int n = 0; n < parent_layer->number_maps; n++) {
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

														node_index[1] = n * parent_layer->map_size + o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

														index.prev_node = node_index[1];
														index.next_node = node_index[0];
														index.weight = weight_index.find(j * parent_layer->number_maps * kernel_size + n * kernel_size + distance[0] * kernel_height * kernel_width + distance[1] * kernel_width + distance[2])->second;

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
	}
}
Connection::~Connection() {
	if (from_error) {
		delete[] from_error;
	}
	if (from_neuron) {
		delete[] from_neuron;
	}
	if (from_weight) {
		delete[] from_weight;
	}
	if (optimizer) {
		delete optimizer;
	}
	delete[] weight;
}

void Connection::Initialize(double scale) {
	for (int i = 0; i < number_weights; i++) {
		weight[i] = scale * (2.0 * rand() / RAND_MAX - 1);
	}
}


void Layer::Construct(int number_maps, int map_width, int map_height, int map_depth, int activation, string properties) {
	this->activation = activation;
	this->batch_size = 1;
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->map_size = map_depth * map_height * map_width;
	this->number_maps = number_maps;
	this->number_nodes = number_maps * map_depth * map_height * map_width;
	this->optimizer = nullptr;
	this->properties = properties;

	memset(bias = new float[number_maps], 0, sizeof(float) * number_maps);
	error = new float[number_nodes];
	neuron = new float[number_nodes];
}

Layer::Layer(int number_nodes, int activation, string properties) {
	Construct(number_nodes, 1, 1, 1, activation, properties);
}
Layer::Layer(int number_maps, int map_width, int activation, string prperties) {
	Construct(number_maps, map_width, 1, 1, activation, properties);
}
Layer::Layer(int number_maps, int map_width, int map_height, int activation, string properties) {
	Construct(number_maps, map_width, map_height, 1, activation, properties);
}
Layer::Layer(int number_maps, int map_width, int map_height, int map_depth, int activation, string properties) {
	Construct(number_maps, map_width, map_height, map_depth, activation, properties);
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

		for (int j = 0; j < number_nodes; j++) {
			double sum = 0;

			for (int k = 0; k < child_connection.size(); k++) {
				Connection *connection = child_connection[k];

				Layer *child_layer = connection->layer;

				float *next_error = &child_layer->error[h * child_layer->number_nodes];

				vector<Index> &from_error = connection->from_error[j];

				for (auto index = from_error.begin(); index != from_error.end(); index++) {
					sum += next_error[index->next_node] * connection->weight[index->weight];
				}
			}
			error[j] = sum;
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

				vector<Index> &from_neuron = connection->from_neuron[j];

				for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
					sum += prev_neuron[index->prev_node] * connection->weight[index->weight];
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


Optimizer::Optimizer(int type, double learning_rate, double momentum, int number_parameters) {
	this->gradient = nullptr;
	this->type = type;
	this->learning_rate = learning_rate;
	this->momentum = momentum;

	if (type) {
		memset(gradient = new float[number_parameters], 0, sizeof(float) * number_parameters);
	}
}
Optimizer::~Optimizer() {
	if (gradient) {
		delete[] gradient;
	}
}

double Optimizer::Calculate_Gradient(int index, double gradient) {
	double output = 0;

	switch (type) {
	case SGD:
		output = -learning_rate * gradient;
		break;
	case Momentum:
		output = this->gradient[index] * momentum - learning_rate * gradient;
		this->gradient[index] = output;
		break;
	case Nesterov:
		output = (this->gradient[index] * momentum - learning_rate * gradient) * momentum - learning_rate * gradient;
		this->gradient[index] = output;
		break;
	}
	return output;
}

Optimizer* Optimizer::Copy(int number_parameters) {
	return new Optimizer(type, learning_rate, momentum, number_parameters);
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
		layer[i]->optimizer = optimizer->Copy(layer[i]->number_maps);
	}
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

			// prepare dropout mask if specified
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
			for (int i = layer.size() - 2; i > 0; i--) {
				layer[i]->Backward();
				layer[i]->Derivative();
			}

			// adjust bias
			for (int i = 1; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				#pragma omp parallel for
				for (int j = 0; j < layer->number_maps; j++) {
					double sum = 0;

					for (int h = 0; h < this->batch_size; h++) {
						sum += layer->error[h * layer->number_nodes + j * layer->map_size];
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
					double sum = 0;

					vector<Index> &from_weight = connection->from_weight[j];

					for (int h = 0; h < this->batch_size; h++) {
						float *error = &layer->error[h * layer->number_nodes];
						float *neuron = &parent_layer->neuron[h * parent_layer->number_nodes];

						for (auto index = from_weight.begin(); index != from_weight.end(); index++) {
							sum += error[index->next_node] * neuron[index->prev_node];
						}
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

Connection* Neural_Networks::Connect(int from, int to, string properties) {
	Connection *connection = new Connection(layer[from], layer[to], properties);

	this->connection.push_back(connection);
	layer[from]->connection.push_back(connection);
	layer[to]->child_connection.push_back(connection);

	return connection;
}
