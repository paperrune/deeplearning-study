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

__device__ void Activate(float *neuron, int index, int activation, int number_nodes) {
	if (activation == Activation::softmax) {
		if (index % number_nodes == 0) {
			double max;
			double sum = 0;

			for (int j = 0; j < number_nodes; j++) {
				if (j == 0 || max < neuron[index + j]) {
					max = neuron[index + j];
				}
			}
			for (int j = 0; j < number_nodes; j++) {
				sum += (neuron[index + j] = exp(neuron[index + j] - max));
			}
			for (int j = 0; j < number_nodes; j++) {
				neuron[index + j] /= sum;
			}
		}
	}
	else if (activation == Activation::linear) {
		// neuron[index] = neuron[index];
	}
	else if (activation == Activation::hard_sigmoid) {
		double slope = 0.2;
		double shift = 0.5;

		neuron[index] = neuron[index] * slope + shift;
		neuron[index] = (1 < neuron[index]) ? (1) : ((0 > neuron[index]) ? (0) : (neuron[index]));
	}
	else if (activation == Activation::relu) {
		neuron[index] *= (neuron[index] > 0);
	}
	else if (activation == Activation::sigmoid) {
		neuron[index] = 1 / (1 + exp(-neuron[index]));
	}
	else if (activation == Activation::tanh) {
		neuron[index] = 2 / (1 + exp(-2 * neuron[index])) - 1;
	}
}
__device__ void Differentiate(float *error, float *neuron, int index, int activation, int batch_size, int number_nodes, double scale = 1, int loss = -1, float y_data[] = nullptr) {
	if (y_data) {
		if (loss == Loss::cross_entropy) {
			if (activation == Activation::sigmoid) {
				error[index] = (neuron[index] - y_data[index]) / (number_nodes * batch_size);
			}
			else if (activation == Activation::softmax) {
				error[index] = (neuron[index] - y_data[index]) / batch_size;
			}
		}
		else if (loss == Loss::mean_squared_error) {
			error[index] = 2 * (neuron[index] - y_data[index]) / (number_nodes * batch_size);
		}
	}
	else {
		double scaled_neuron = neuron[index] * scale;

		if (activation == Activation::linear) {
			// error[index] *= 1;
		}
		else if (activation == Activation::hard_sigmoid) {
			double slope = 0.2;
			double shift = 0.5;

			error[index] = (neuron[index] == 0 || neuron[index] == 1) ? (0) : (slope);
		}
		else if (activation == Activation::relu) {
			error[index] = (neuron[index] > 0) ? (error[index]) : (0);
		}
		else if (activation == Activation::sigmoid && loss != Loss::cross_entropy) {
			error[index] *= (1 - scaled_neuron) * scaled_neuron;
		}
		else if (activation == Activation::tanh) {
			error[index] *= (1 - scaled_neuron) * (1 + scaled_neuron);
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
__device__ double Derivation(double x, int activation) {
	double y = 1;

	if (activation == Activation::linear) {
		// y = 1
	}
	else if (activation == Activation::hard_sigmoid) {
		double slope = 0.2;
		double shift = 0.5;

		y = (x == 0 || x == 1) ? (0) : (slope);
	}
	else if (activation == Activation::relu) {
		y = (x > 0);
	}
	else if (activation == Activation::sigmoid) {
		y = (1 - x) * x;
	}
	else if (activation == Activation::tanh) {
		y = (1 - x) * (1 + x);
	}
	return y;
}

__global__ void Activate(Batch_Normalization batch_normalization, int time_index, float _neuron[], bool training) {
	int j = blockIdx.x;
	int t = time_index;

	float *mean = &batch_normalization.mean[t * batch_normalization.number_maps];
	float *variance = &batch_normalization.variance[t * batch_normalization.number_maps];
	float *moving_mean = &batch_normalization.moving_mean[t * batch_normalization.number_maps];
	float *moving_variance = &batch_normalization.moving_variance[t * batch_normalization.number_maps];

	if (training) {
		float *neuron = &_neuron[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
		float *neuron_normalized = &batch_normalization.neuron_normalized[t * batch_normalization.number_nodes + j * batch_normalization.map_size];

		__shared__ double standard_deviation;
		__shared__ double sum[NUMBER_THREADS];

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

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
			moving_mean[j] = batch_normalization.momentum * moving_mean[j] + (1 - batch_normalization.momentum) * (mean[j] = sum[0] / (batch_normalization.batch_size * batch_normalization.map_size));
		}

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				sum[threadIdx.x] += (neuron[index + k] - mean[j]) * (neuron[index + k] - mean[j]);
			}
		}
		for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
			__syncthreads();

			if (threadIdx.x < h) {
				sum[threadIdx.x] += sum[threadIdx.x + h];
			}
		}
		if (threadIdx.x == 0) {
			moving_variance[j] = batch_normalization.momentum * moving_variance[j] + (1 - batch_normalization.momentum) * (variance[j] = sum[0] / (batch_normalization.batch_size * batch_normalization.map_size));
			standard_deviation = sqrt(variance[j] + batch_normalization.epsilon);
		}
		__syncthreads();

		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				neuron_backup[index + k] = neuron[index + k];
				neuron_normalized[index + k] = (neuron[index + k] - mean[j]) / standard_deviation;
				neuron[index + k] = batch_normalization.gamma[j] * neuron_normalized[index + k] + batch_normalization.beta[j];
			}
		}
	}
	else {
		float *neuron = &_neuron[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[t * batch_normalization.number_nodes + j * batch_normalization.map_size];

		double standard_deviation = sqrt(moving_variance[j] + batch_normalization.epsilon);

		for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
			int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

			for (int k = 0; k < batch_normalization.map_size; k++) {
				neuron_backup[index + k] = neuron[index + k];
				neuron[index + k] = batch_normalization.gamma[j] / standard_deviation * neuron[index + k] + (batch_normalization.beta[j] - batch_normalization.gamma[j] * moving_mean[j] / standard_deviation);
			}
		}
	}
}
__global__ void Activate(Layer layer, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		Activate(layer.neuron, (blockIdx.x * layer.time_step + t) * layer.number_nodes + j, layer.activation, layer.number_nodes);
	}
}
__global__ void Activate(Layer layer, int time_index, Dropout dropout, bool training) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		int index = (blockIdx.x * layer.time_step + t) * layer.number_nodes + j;

		if (training) {
			if (dropout.mask[index]) {
				layer.neuron[index] /= (1 - dropout.rate);
			}
			else {
				layer.neuron[index] = 0;
			}
		}
		Activate(layer.neuron, index, layer.activation, layer.number_nodes);
	}
}
__global__ void Activate(LSTM lstm, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		int index = (blockIdx.x * lstm.time_step + t) * lstm.number_nodes + j;

		float *forget_neuron[] = { lstm.neuron[LSTM::forget][0], lstm.neuron[LSTM::forget][1] };
		float *input_neuron[] = { lstm.neuron[LSTM::input][0], lstm.neuron[LSTM::input][1] };
		float *output_neuron[] = { lstm.neuron[LSTM::output][0], lstm.neuron[LSTM::output][1] };
		float *cell_neuron[] = { lstm.neuron[LSTM::cell][0], lstm.neuron[LSTM::cell][1] };
		float *cell_output_neuron[] = { lstm.neuron[LSTM::cell_output][0], lstm.neuron[LSTM::cell_output][1] };

		float *previous_cell_output_neuron = ((lstm.direction == 1 && t > 0) || (lstm.direction == -1 && t < lstm.time_step - 1)) ? (&lstm.neuron[LSTM::cell_output][1][(blockIdx.x * lstm.time_step + t - lstm.direction) * lstm.number_nodes]) : (nullptr);

		forget_neuron[0][index] += (forget_neuron[1][index] + lstm.bias[LSTM::forget][j / lstm.map_size]);
		input_neuron[0][index] += (input_neuron[1][index] + lstm.bias[LSTM::input][j / lstm.map_size]);
		output_neuron[0][index] += (output_neuron[1][index] + lstm.bias[LSTM::output][j / lstm.map_size]);
		cell_neuron[0][index] += (cell_neuron[1][index] + lstm.bias[LSTM::cell][j / lstm.map_size]);

		Activate(forget_neuron[0], index, lstm.recurrent_activation, lstm.number_nodes);
		Activate(input_neuron[0], index, lstm.recurrent_activation, lstm.number_nodes);
		Activate(output_neuron[0], index, lstm.recurrent_activation, lstm.number_nodes);
		Activate(cell_neuron[0], index, lstm.activation, lstm.number_nodes);

		cell_output_neuron[0][index] = (previous_cell_output_neuron) ? (forget_neuron[0][index] * previous_cell_output_neuron[j] + input_neuron[0][index] * cell_neuron[0][index]) : (input_neuron[0][index] * cell_neuron[0][index]);
		cell_output_neuron[1][index] = cell_output_neuron[0][index];
	}
}
__global__ void Activate(LSTM lstm, Layer layer, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		int index = (blockIdx.x * lstm.time_step + t) * lstm.number_nodes + j;

		float *output_neuron = lstm.neuron[LSTM::output][0];
		float *cell_output_neuron = lstm.neuron[LSTM::cell_output][0];

		Activate(cell_output_neuron, index, Activation::tanh, lstm.number_nodes);
		layer.neuron[index] = output_neuron[index] * cell_output_neuron[index];
	}
}
__global__ void Activate(RNN rnn, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		Activate(rnn.neuron[0], (blockIdx.x * rnn.time_step + t) * rnn.number_nodes + j, rnn.activation, rnn.number_nodes);
	}
}
__global__ void Add_Bias(Layer layer, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		layer.neuron[(blockIdx.x * layer.time_step + t) * layer.number_nodes + j] += layer.bias[j / layer.map_size];
	}
}
__global__ void Add_Bias(RNN rnn, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		int index = (blockIdx.x * rnn.time_step + t) * rnn.number_nodes + j;

		rnn.neuron[0][index] += (rnn.neuron[1][index] + rnn.bias[j / rnn.map_size]);
	}
}
__global__ void Adjust_Bias(Layer layer, Optimizer optimizer, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < layer.number_maps) {
		double sum = 0;

		for (int h = 0; h < layer.batch_size * layer.time_step; h++) {
			for (int k = 0; k < layer.map_size; k++) {
				sum += layer.error[h * layer.number_nodes + j * layer.map_size + k];
			}
		}
		layer.bias[j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Bias(LSTM lstm, Optimizer optimizer, int type, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < lstm.number_maps) {
		double sum = 0;

		for (int h = 0; h < lstm.batch_size * lstm.time_step; h++) {
			for (int k = 0; k < lstm.map_size; k++) {
				sum += lstm.error[type][0][h * lstm.number_nodes + j * lstm.map_size + k];
			}
		}
		lstm.bias[type][j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Bias(LSTM lstm, Optimizer optimizer, Batch_Normalization batch_normalization, int type, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < lstm.number_maps) {
		double sum = 0;

		for (int h = 0; h < lstm.batch_size * lstm.time_step; h++) {
			for (int k = 0; k < lstm.map_size; k++) {
				sum += batch_normalization.error_backup[h * lstm.number_nodes + j * lstm.map_size + k];
			}
		}
		lstm.bias[type][j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Bias(RNN rnn, Optimizer optimizer, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < rnn.number_maps) {
		double sum = 0;

		for (int h = 0; h < rnn.batch_size * rnn.time_step; h++) {
			for (int k = 0; k < rnn.map_size; k++) {
				sum += rnn.error[0][h * rnn.number_nodes + j * rnn.map_size + k];
			}
		}
		rnn.bias[j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Bias(RNN rnn, Optimizer optimizer, Batch_Normalization batch_normalization, int iterations) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < rnn.number_maps) {
		double sum = 0;

		for (int h = 0; h < rnn.batch_size * rnn.time_step; h++) {
			for (int k = 0; k < rnn.map_size; k++) {
				sum += batch_normalization.error_backup[h * rnn.number_nodes + j * rnn.map_size + k];
			}
		}
		rnn.bias[j] += Calculate_Gradient(optimizer, j, sum, iterations);
	}
}
__global__ void Adjust_Parameter(Batch_Normalization batch_normalization, Optimizer gamma_optimizer, Optimizer beta_optimizer, int iterations) {
	int j = blockIdx.x;

	float *error_backup = &batch_normalization.error_backup[j * batch_normalization.map_size];
	float *neuron_normalized = &batch_normalization.neuron_normalized[j * batch_normalization.map_size];

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size * batch_normalization.time_step; h += blockDim.x) {
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
	for (int h = threadIdx.x; h < batch_normalization.batch_size * batch_normalization.time_step; h += blockDim.x) {
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
	int l = blockIdx.x * blockDim.x + threadIdx.x;

	if (l < connection.number_weights) {
		double sum = 0;

		for (int t = 0, j = l / connection.number_weights_per_map, k = ((connection.depthwise) ? (j * layer.number_maps / parent_layer.number_maps) : ((l / connection.kernel_size) % parent_layer.number_maps)), m = l % connection.kernel_size; t < layer.time_step; t++) {
			for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += layer.time_step) {
				for (int h = 0; h < layer.batch_size; h++) {
					float *error = &layer.error[(h * layer.time_step + t) * layer.number_nodes + j * layer.map_size];
					float *prev_neuron = &parent_layer.neuron[(h * parent_layer.time_step + (*s)) * parent_layer.number_nodes + k * parent_layer.map_size];

					for (Index *index = &connection.from_weight_device[m]; index->weight != -1; index += connection.kernel_size) {
						sum += error[index->next_node] * prev_neuron[index->prev_node];
					}
				}
			}
		}
		connection.weight[l] += Calculate_Gradient(optimizer, l, sum, iterations);
	}
}
__global__ void Adjust_Weight(LSTM lstm, Layer layer, Connection connection, Optimizer optimizer, int iterations, bool recurrent = false) {
	int l = blockIdx.x * blockDim.x + threadIdx.x;

	if (l < connection.number_weights) {
		double sum = 0;

		if (recurrent) {
			for (int h = 0, j = l / connection.number_weights_per_map, k = (l / connection.kernel_size) % lstm.number_maps, m = l % connection.kernel_size; h < lstm.batch_size * lstm.time_step; h++) {
				if ((lstm.direction == 1 && h % lstm.time_step > 0) || (lstm.direction == -1 && h % lstm.time_step < lstm.time_step - 1)) {
					float *error = &lstm.error[connection.type][1][h * lstm.number_nodes + j * lstm.map_size];
					float *neuron = &layer.neuron[(h - lstm.direction) * lstm.number_nodes + k * layer.map_size];

					for (Index *index = &connection.from_weight_device[m]; index->weight != -1; index += connection.kernel_size) {
						sum += error[index->next_node] * neuron[index->prev_node];
					}
				}
			}
		}
		else {
			for (int t = 0, j = l / connection.number_weights_per_map, k = (l / connection.kernel_size) % layer.number_maps, m = l % connection.kernel_size; t < lstm.time_step; t++) {
				for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += lstm.time_step) {
					for (int h = 0; h < lstm.batch_size; h++) {
						float *error = &lstm.error[connection.type][0][(h * lstm.time_step + t) * lstm.number_nodes + j * lstm.map_size];
						float *neuron = &layer.neuron[(h * layer.time_step + (*s)) * layer.number_nodes + k * lstm.map_size];

						for (Index *index = &connection.from_weight_device[m]; index->weight != -1; index += connection.kernel_size) {
							sum += error[index->next_node] * neuron[index->prev_node];
						}
					}
				}
			}
		}
		connection.weight[l] += Calculate_Gradient(optimizer, l, sum, iterations);
	}
}
__global__ void Adjust_Weight(RNN rnn, Connection connection, Optimizer optimizer, int iterations) {
	int l = blockIdx.x * blockDim.x + threadIdx.x;

	if (l < connection.number_weights) {
		double sum = 0;

		for (int h = 0, j = l / connection.number_weights_per_map, k = (l / connection.kernel_size) % rnn.number_maps, m = l % connection.kernel_size; h < rnn.batch_size * rnn.time_step; h++) {
			if ((rnn.direction == 1 && h % rnn.time_step > 0) || (rnn.direction == -1 && h % rnn.time_step < rnn.time_step - 1)) {
				float *error = &rnn.error[1][h * rnn.number_nodes + j * rnn.map_size];
				float *neuron = &rnn.neuron[0][(h - rnn.direction) * rnn.number_nodes + k * rnn.map_size];

				for (Index *index = &connection.from_weight_device[m]; index->weight != -1; index += connection.kernel_size) {
					sum += error[index->next_node] * neuron[index->prev_node];
				}
			}
		}
		connection.weight[l] += Calculate_Gradient(optimizer, l, sum, iterations);
	}
}
__global__ void Adjust_Weight(RNN rnn, Layer parent_layer, Connection connection, Optimizer optimizer, int iterations) {
	int l = blockIdx.x * blockDim.x + threadIdx.x;

	if (l < connection.number_weights) {
		double sum = 0;

		for (int t = 0, j = l / connection.number_weights_per_map, k = (l / connection.kernel_size) % parent_layer.number_maps, m = l % connection.kernel_size; t < rnn.time_step; t++) {
			for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += rnn.time_step) {
				for (int h = 0; h < rnn.batch_size; h++) {
					float *error = &rnn.error[0][(h * rnn.time_step + t) * rnn.number_nodes + j * rnn.map_size];
					float *neuron = &parent_layer.neuron[(h * parent_layer.time_step + (*s)) * parent_layer.number_nodes + k * parent_layer.map_size];

					for (Index *index = &connection.from_weight_device[m]; index->weight != -1; index += connection.kernel_size) {
						sum += error[index->next_node] * neuron[index->prev_node];
					}
				}
			}
		}
		connection.weight[l] += Calculate_Gradient(optimizer, l, sum, iterations);
	}
}
__global__ void Backward(Layer layer, Layer parent_layer, Connection connection, int time_index, int type) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < parent_layer.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t], k = j / parent_layer.map_size, l = j % parent_layer.map_size; *s != -1; s += layer.time_step) {
			float *error = &layer.error[(blockIdx.x * layer.time_step + t) * layer.number_nodes];
			float *prev_error = &parent_layer.error[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			double sum = 0;

			if (type == 0) { // average-pooling
				int offset = k * layer.map_size;

				for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += parent_layer.map_size) {
					sum += error[offset + index->next_node] / connection.from_neuron_device[index->next_node].weight;
				}
				prev_error[j] += sum;
			}
			else if (type == 1) { // max-pooling
				int offset = k * layer.map_size;

				float *neuron = &layer.neuron[(blockIdx.x * layer.time_step + t) * layer.number_nodes + offset];
				float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

				for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += parent_layer.map_size) {
					if (prev_neuron[j] == neuron[index->next_node]) {
						sum += error[offset + index->next_node];
					}
				}
				prev_error[j] += sum;
			}
			else if (type == 2) {
				for (int *m = &connection.channel_connection_device[1][k]; *m != -1; m += parent_layer.number_maps) {
					int offset[] = { (*m) * layer.map_size, (*m) * connection.number_weights_per_map + ((connection.depthwise) ? (0) : (k * connection.kernel_size)) };

					for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += parent_layer.map_size) {
						sum += error[offset[0] + index->next_node] * connection.weight[offset[1] + index->weight];
					}
				}
				prev_error[j] += sum;
			}
		}
	}
}
__global__ void Backward(LSTM lstm, Layer layer, Connection connection, int time_index, bool recurrent = false) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if ((recurrent && j < lstm.number_nodes) || j < layer.number_nodes) {
		if (recurrent) {
			float *error = &lstm.error[connection.type][1][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
			float *prev_error = &layer.error[(blockIdx.x * lstm.time_step + t - lstm.direction) * lstm.number_nodes];

			double sum = 0;

			for (int k = j / lstm.map_size, l = j % lstm.map_size, *m = &connection.channel_connection_device[1][k]; *m != -1; m += lstm.number_maps) {
				int offset[] = { (*m) * lstm.map_size, (*m) * connection.number_weights_per_map + k * connection.kernel_size };

				for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += lstm.map_size) {
					sum += error[offset[0] + index->next_node] * connection.weight[offset[1] + index->weight];
				}
			}
			prev_error[j] += sum;
		}
		else {
			for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += lstm.time_step) {
				float *error = &lstm.error[connection.type][0][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
				float *prev_error = &layer.error[(blockIdx.x * layer.time_step + (*s)) * layer.number_nodes];

				double sum = 0;

				for (int k = j / layer.map_size, l = j % layer.map_size, *m = &connection.channel_connection_device[1][k]; *m != -1; m += layer.number_maps) {
					int offset[] = { (*m) * lstm.map_size, (*m) * connection.number_weights_per_map + k * connection.kernel_size };

					for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += layer.map_size) {
						sum += error[offset[0] + index->next_node] * connection.weight[offset[1] + index->weight];
					}
				}
				prev_error[j] += sum;
			}
		}
	}
}
__global__ void Backward(RNN rnn, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		float *error = &rnn.error[1][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
		float *prev_error = &rnn.error[0][(blockIdx.x * rnn.time_step + t - rnn.direction) * rnn.number_nodes];

		double sum = 0;

		for (int k = j / rnn.map_size, l = j % rnn.map_size, *m = &connection.channel_connection_device[1][k]; *m != -1; m += rnn.number_maps) {
			int offset[] = { (*m) * rnn.map_size, (*m) * connection.number_weights_per_map + k * connection.kernel_size };

			for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += rnn.map_size) {
				sum += error[offset[0] + index->next_node] * connection.weight[offset[1] + index->weight];
			}
		}
		prev_error[j] += sum;
	}
}
__global__ void Backward(RNN rnn, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < parent_layer.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += rnn.time_step) {
			float *error = &rnn.error[0][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
			float *prev_error = &parent_layer.error[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			double sum = 0;

			for (int k = j / parent_layer.map_size, l = j % parent_layer.map_size, *m = &connection.channel_connection_device[1][k]; *m != -1; m += parent_layer.number_maps) {
				int offset[] = { (*m) * rnn.map_size, (*m) * connection.number_weights_per_map + k * connection.kernel_size };

				for (Index *index = &connection.from_error_device[l]; index->weight != -1; index += parent_layer.map_size) {
					sum += error[offset[0] + index->next_node] * connection.weight[offset[1] + index->weight];
				}
			}
			prev_error[j] += sum;
		}
	}
}
__global__ void Calculate_Loss(Layer layer, bool time_mask[], int loss, float y_data[]) {
	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;

	for (int j = threadIdx.x; j < layer.batch_size * layer.time_step * layer.number_nodes; j += blockDim.x) {
		if (time_mask == nullptr || time_mask[(j / layer.number_nodes) % layer.time_step]) {
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
__global__ void Copy_Backward(Layer layer, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += layer.time_step) {
			float *error = &layer.error[(blockIdx.x * layer.time_step + t) * layer.number_nodes];
			float *prev_error = &parent_layer.error[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			prev_error[j] = error[j];
		}
	}
}
__global__ void Copy_Backward(LSTM lstm, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += lstm.time_step) {
			float *error = &lstm.error[connection.type][0][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
			float *prev_error = &parent_layer.error[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			prev_error[j] = error[j];
		}
	}
}
__global__ void Copy_Backward(RNN rnn, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += rnn.time_step) {
			float *error = &rnn.error[0][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
			float *prev_error = &parent_layer.error[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			prev_error[j] = error[j];
		}
	}
}
__global__ void Copy_Forward(Layer layer, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += layer.time_step) {
			float *neuron = &layer.neuron[(blockIdx.x * layer.time_step + t) * layer.number_nodes];
			float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			neuron[j] = prev_neuron[j];
		}
	}
}
__global__ void Copy_Forward(LSTM lstm, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += lstm.time_step) {
			float *neuron = &lstm.neuron[connection.type][0][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
			float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			neuron[j] = prev_neuron[j];
		}
	}
}
__global__ void Copy_Forward(RNN rnn, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += rnn.time_step) {
			float *neuron = &rnn.neuron[0][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
			float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			neuron[j] = prev_neuron[j];
		}
	}
}
__global__ void Differentiate(Batch_Normalization batch_normalization, int time_index, float _error[]) {
	int j = blockIdx.x;
	int t = time_index;

	float *mean = &batch_normalization.mean[t * batch_normalization.number_maps];
	float *variance = &batch_normalization.variance[t * batch_normalization.number_maps];

	float *error = &_error[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
	float *error_backup = &batch_normalization.error_backup[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
	float *error_normalized = &batch_normalization.error_normalized[t * batch_normalization.number_nodes + j * batch_normalization.map_size];
	float *neuron_backup = &batch_normalization.neuron_backup[t * batch_normalization.number_nodes + j * batch_normalization.map_size];

	double standard_deviation = sqrt(variance[j] + batch_normalization.epsilon);

	__shared__ double error_mean;
	__shared__ double error_variance;
	__shared__ double sum[2][NUMBER_THREADS];

	sum[0][threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			error_normalized[index + k] = error[index + k] * batch_normalization.gamma[j];
			sum[0][threadIdx.x] += error_normalized[index + k] * (neuron_backup[index + k] - mean[j]);
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[0][threadIdx.x] += sum[0][threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		error_variance = sum[0][threadIdx.x] * (-0.5) * pow(variance[j] + batch_normalization.epsilon, -1.5);
	}

	sum[0][threadIdx.x] = 0;
	sum[1][threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_normalization.batch_size; h += blockDim.x) {
		int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			sum[0][threadIdx.x] += error_normalized[index + k];
			sum[1][threadIdx.x] += (neuron_backup[index + k] - mean[j]);
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
		int index = h * batch_normalization.time_step * batch_normalization.number_nodes;

		for (int k = 0; k < batch_normalization.map_size; k++) {
			error_backup[index + k] = error[index + k];
			error[index + k] = error_normalized[index + k] / standard_deviation + error_variance * 2 * (neuron_backup[index + k] - mean[j]) / (batch_normalization.batch_size * batch_normalization.map_size) + error_mean / (batch_normalization.batch_size * batch_normalization.map_size);
		}
	}
}
__global__ void Differentiate(Layer layer, int time_index, int loss = -1, float y_data[] = nullptr) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		Differentiate(layer.error, layer.neuron, (blockIdx.x * layer.time_step + time_index) * layer.number_nodes + j, layer.activation, layer.batch_size, layer.number_nodes, 1, loss, y_data);
	}
}
__global__ void Differentiate(Layer layer, Dropout dropout, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < layer.number_nodes) {
		int index = (blockIdx.x * layer.time_step + time_index) * layer.number_nodes + j;

		if (dropout.mask[index]) {
			Differentiate(layer.error, layer.neuron, index, layer.activation, layer.batch_size, layer.number_nodes, 1 - dropout.rate);
			layer.error[index] /= (1 - dropout.rate);
		}
		else {
			layer.error[index] = 0;
		}
	}
}
__global__ void DIfferentiate(LSTM lstm, Layer layer, int time_index, int type) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		int index = (blockIdx.x * lstm.time_step + t) * lstm.number_nodes + j;

		if (type == 0) {
			float *error = layer.error;
			float *cell_output_error[] = { lstm.error[LSTM::cell_output][0], lstm.error[LSTM::cell_output][1] };
			float *previous_cell_output_error = ((lstm.direction == 1 && t > 0) || (lstm.direction == -1 && t < lstm.time_step - 1)) ? (&lstm.error[LSTM::cell_output][0][(blockIdx.x * lstm.time_step + t - lstm.direction) * lstm.number_nodes]) : (nullptr);

			float *forget_neuron[] = { lstm.neuron[LSTM::forget][0], lstm.neuron[LSTM::forget][1] };
			float *output_neuron[] = { lstm.neuron[LSTM::output][0], lstm.neuron[LSTM::output][1] };
			float *cell_output_neuron[] = { lstm.neuron[LSTM::cell_output][0], lstm.neuron[LSTM::cell_output][1] };

			cell_output_error[0][index] += error[index] * output_neuron[0][index] * Derivation(cell_output_neuron[0][index], Activation::tanh);

			if (previous_cell_output_error) {
				previous_cell_output_error[j] = forget_neuron[0][index] * cell_output_error[0][index];
			}
		}
		else {
			float *error = layer.error;
			float *forget_error[] = { lstm.error[LSTM::forget][0], lstm.error[LSTM::forget][1] };
			float *input_error[] = { lstm.error[LSTM::input][0], lstm.error[LSTM::input][1] };
			float *output_error[] = { lstm.error[LSTM::output][0], lstm.error[LSTM::output][1] };
			float *cell_error[] = { lstm.error[LSTM::cell][0], lstm.error[LSTM::cell][1] };
			float *cell_output_error[] = { lstm.error[LSTM::cell_output][0], lstm.error[LSTM::cell_output][1] };

			float *forget_neuron[] = { lstm.neuron[LSTM::forget][0], lstm.neuron[LSTM::forget][1] };
			float *input_neuron[] = { lstm.neuron[LSTM::input][0], lstm.neuron[LSTM::input][1] };
			float *output_neuron[] = { lstm.neuron[LSTM::output][0], lstm.neuron[LSTM::output][1] };
			float *cell_neuron[] = { lstm.neuron[LSTM::cell][0], lstm.neuron[LSTM::cell][1] };
			float *cell_output_neuron[] = { lstm.neuron[LSTM::cell_output][0], lstm.neuron[LSTM::cell_output][1] };

			float *previous_cell_output_neuron = ((lstm.direction == 1 && t > 0) || (lstm.direction == -1 && t < lstm.time_step - 1)) ? (&lstm.neuron[LSTM::cell_output][1][(blockIdx.x * lstm.time_step + t - lstm.direction) * lstm.number_nodes]) : (nullptr);

			forget_error[0][index] = (previous_cell_output_neuron) ? (cell_output_error[0][index] * previous_cell_output_neuron[j] * Derivation(forget_neuron[0][index], lstm.recurrent_activation)) : (0);
			forget_error[1][index] = forget_error[0][index];

			input_error[0][index] = cell_output_error[0][index] * cell_neuron[0][index] * Derivation(input_neuron[0][index], lstm.recurrent_activation);
			input_error[1][index] = input_error[0][index];

			output_error[0][index] = error[index] * cell_output_neuron[0][index] * Derivation(output_neuron[0][index], lstm.recurrent_activation);
			output_error[1][index] = output_error[0][index];

			cell_error[0][index] = cell_output_error[0][index] * input_neuron[0][index] * Derivation(cell_neuron[0][index], lstm.activation);
			cell_error[1][index] = cell_error[0][index];
		}
	}
}
__global__ void Differentiate(RNN rnn, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;

	if (j < rnn.number_nodes) {
		int index = (blockIdx.x * rnn.time_step + time_index) * rnn.number_nodes + j;

		Differentiate(rnn.error[0], rnn.neuron[0], index, rnn.activation, rnn.batch_size, rnn.number_nodes);
		rnn.error[1][index] = rnn.error[0][index];
	}
}
__global__ void Forward(Layer layer, Layer parent_layer, Connection connection, int time_index, int type) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < layer.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t], k = j / layer.map_size, l = j % layer.map_size; *s != -1; s += layer.time_step) {
			float *neuron = &layer.neuron[(blockIdx.x * layer.time_step + t) * layer.number_nodes];
			float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			double sum = 0;

			if (type == 0) { // average-pooling
				int number_connections = 0;
				int offset = k * parent_layer.map_size;

				double sum = 0;

				for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += layer.map_size, number_connections++) {
					sum += prev_neuron[offset + index->prev_node];
				}
				neuron[j] += sum / number_connections;
			}
			else if (type == 1) { // max-pooling
				int offset = k * parent_layer.map_size;

				for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += layer.map_size) {
					if (index == &connection.from_neuron_device[l] || sum < prev_neuron[offset + index->prev_node]) {
						sum = prev_neuron[offset + index->prev_node];
					}
				}
				neuron[j] += sum;
			}
			else if (type == 2) {
				for (int *m = &connection.channel_connection_device[0][k]; *m != -1; m += layer.number_maps) {
					int offset[] = { (*m) * parent_layer.map_size, k * connection.number_weights_per_map + ((connection.depthwise) ? (0) : ((*m) * connection.kernel_size)) };

					for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += layer.map_size) {
						sum += prev_neuron[offset[0] + index->prev_node] * connection.weight[offset[1] + index->weight];
					}
				}
				neuron[j] += sum;
			}
		}
	}
}
__global__ void Forward(LSTM lstm, Layer layer, Connection connection, int time_index, bool recurrent = false) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < lstm.number_nodes) {
		if (recurrent) {
			float *neuron = &lstm.neuron[connection.type][1][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
			float *prev_neuron = &layer.neuron[(blockIdx.x * lstm.time_step + t - lstm.direction) * lstm.number_nodes];

			double sum = 0;

			for (int k = j / lstm.map_size, l = j % lstm.map_size, *m = &connection.channel_connection_device[0][k]; *m != -1; m += lstm.number_maps) {
				int offset[] = { (*m) * lstm.map_size, k * connection.number_weights_per_map + (*m) * connection.kernel_size };

				for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += lstm.map_size) {
					sum += prev_neuron[offset[0] + index->prev_node] * connection.weight[offset[1] + index->weight];
				}
			}
			neuron[j] += sum;
		}
		else {
			for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += lstm.time_step) {
				float *neuron = &lstm.neuron[connection.type][0][(blockIdx.x * lstm.time_step + t) * lstm.number_nodes];
				float *prev_neuron = &layer.neuron[(blockIdx.x * layer.time_step + (*s)) * layer.number_nodes];

				double sum = 0;

				for (int k = j / lstm.map_size, l = j % lstm.map_size, *m = &connection.channel_connection_device[0][k]; *m != -1; m += lstm.number_maps) {
					int offset[] = { (*m) * layer.map_size, k * connection.number_weights_per_map + (*m) * connection.kernel_size };

					for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += lstm.map_size) {
						sum += prev_neuron[offset[0] + index->prev_node] * connection.weight[offset[1] + index->weight];
					}
				}
				neuron[j] += sum;
			}
		}
	}
}
__global__ void Forward(RNN rnn, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		float *neuron = &rnn.neuron[1][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
		float *prev_neuron = &rnn.neuron[0][(blockIdx.x * rnn.time_step + t - rnn.direction) * rnn.number_nodes];

		double sum = 0;

		for (int k = j / rnn.map_size, l = j % rnn.map_size, *m = &connection.channel_connection_device[0][k]; *m != -1; m += rnn.number_maps) {
			int offset[] = { (*m) * rnn.map_size, k * connection.number_weights_per_map + (*m) * connection.kernel_size };

			for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += rnn.map_size) {
				sum += prev_neuron[offset[0] + index->prev_node] * connection.weight[offset[1] + index->weight];
			}
		}
		neuron[j] += sum;
	}
}
__global__ void Forward(RNN rnn, Layer parent_layer, Connection connection, int time_index) {
	int j = blockIdx.y * blockDim.x + threadIdx.x;
	int t = time_index;

	if (j < rnn.number_nodes) {
		for (int *s = &connection.time_connection_device[0][t]; *s != -1; s += rnn.time_step) {
			float *neuron = &rnn.neuron[0][(blockIdx.x * rnn.time_step + t) * rnn.number_nodes];
			float *prev_neuron = &parent_layer.neuron[(blockIdx.x * parent_layer.time_step + (*s)) * parent_layer.number_nodes];

			double sum = 0;

			for (int k = j / rnn.map_size, l = j % rnn.map_size, *m = &connection.channel_connection_device[0][k]; *m != -1; m += rnn.number_maps) {
				int offset[] = { (*m) * parent_layer.map_size, k * connection.number_weights_per_map + (*m) * connection.kernel_size };

				for (Index *index = &connection.from_neuron_device[l]; index->weight != -1; index += rnn.map_size) {
					sum += prev_neuron[offset[0] + index->prev_node] * connection.weight[offset[1] + index->weight];
				}
			}
			neuron[j] += sum;
		}
	}
}
__global__ void Random_Normal(float memory[], int memory_size, double scale, int seed = 0, bool add = false) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		curandState s[NUMBER_THREADS];

		curand_init(seed + j, 0, 0, &s[threadIdx.x]);
		memory[j] = (add) ? (memory[j] + scale * curand_normal(&s[threadIdx.x])) : (scale * curand_normal(&s[threadIdx.x]));
	}
}

Batch_Normalization::Batch_Normalization(int time_step, int number_maps, int map_size, double epsilon, double momentum, Layer *layer) {
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
	this->time_step = time_step;

	cudaMalloc(&beta, sizeof(float) * number_maps);
	cudaMalloc(&gamma, sizeof(float) * number_maps);
	cudaMalloc(&mean, sizeof(float) * time_step * number_maps);
	cudaMalloc(&variance, sizeof(float) * time_step * number_maps);
	cudaMalloc(&moving_mean, sizeof(float) * time_step * number_maps);
	cudaMalloc(&moving_variance, sizeof(float) * time_step * number_maps);

	cudaMalloc(&error_backup, sizeof(float) * time_step * number_nodes);
	cudaMalloc(&error_normalized, sizeof(float) * time_step * number_nodes);
	cudaMalloc(&neuron_backup, sizeof(float) * time_step * number_nodes);
	cudaMalloc(&neuron_normalized, sizeof(float) * time_step * number_nodes);
}
Batch_Normalization::~Batch_Normalization() {}

void Batch_Normalization::Activate(int time_index, float neuron[], bool training) {
	::Activate << <number_maps, NUMBER_THREADS >> > (*this, time_index, neuron, training);
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
void Batch_Normalization::Differentiate(int time_index, float error[]) {
	::Differentiate << <number_maps, NUMBER_THREADS >> > (*this, time_index, error);
}
void Batch_Normalization::Initialize() {
	beta_initializer->Random(number_maps, beta, number_maps, number_maps);
	gamma_initializer->Random(number_maps, gamma, number_maps, number_maps);
	moving_mean_initializer->Random(time_step * number_maps, moving_mean, time_step * number_maps, time_step * number_maps);
	moving_variance_initializer->Random(time_step * number_maps, moving_variance, time_step * number_maps, time_step * number_maps);
}
void Batch_Normalization::Load(ifstream &file) {
	float *memory = new float[time_step * number_maps];

	for (int j = 0; j < number_maps; j++) file >> memory[j];
	cudaMemcpy(gamma, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < number_maps; j++) file >> memory[j];
	cudaMemcpy(beta, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < time_step * number_maps; j++) file >> memory[j];
	cudaMemcpy(moving_mean, memory, sizeof(float) * time_step * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < time_step * number_maps; j++) file >> memory[j];
	cudaMemcpy(moving_variance, memory, sizeof(float) * time_step * number_maps, cudaMemcpyHostToDevice);

	delete[] memory;
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
	int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

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
void Batch_Normalization::Save(ofstream &file) {
	float *memory = new float[time_step * number_maps];

	cudaMemcpy(memory, gamma, sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, beta, sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, moving_mean, sizeof(float) * time_step * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < time_step * number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, moving_variance, sizeof(float) * time_step * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < time_step * number_maps; j++) file << memory[j] << endl;

	delete[] memory;
}

Batch_Normalization* Batch_Normalization::Beta_Initializer(Initializer initializer) {
	if (beta_initializer) {
		delete beta_initializer;
	}
	beta_initializer = initializer.Copy();
	return this;
}
Batch_Normalization* Batch_Normalization::Copy() {
	Batch_Normalization *batch_normalization = new Batch_Normalization(time_step, number_maps, map_size, epsilon, momentum, layer);

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


Connection::Connection(Layer *layer, Layer *parent_layer, string properties, unordered_multimap<int, int> *channel_connection, unordered_multimap<int, int> *time_connection, int type) {
	unordered_map<int, int> weight_index;

	this->channel_connection[0] = new vector<int>[layer->number_maps];
	this->channel_connection[1] = new vector<int>[parent_layer->number_maps];
	this->depthwise = (strstr(properties.c_str(), "depthwise")) ? (true) : (false);
	this->from_error_device = nullptr;
	this->from_neuron_device = nullptr;
	this->from_weight_device = nullptr;
	this->initializer = nullptr;
	this->layer = layer;
	this->parent_layer = parent_layer;
	this->properties = properties;
	this->number_weights = 0;
	this->optimizer = nullptr;
	this->time_connection[0] = new vector<int>[layer->time_step];
	this->time_connection[1] = new vector<int>[parent_layer->time_step];
	this->type = type;
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

	{ // channel_connection
		if (channel_connection == nullptr) {
			if (properties[0] == 'W') {
				for (int j = 0; j < layer->number_maps; j++) {
					for (int k = 0; k < parent_layer->number_maps; k++) {
						if (depthwise == false || j % parent_layer->number_maps == k) {
							this->channel_connection[0][j].push_back(k);
							this->channel_connection[1][k].push_back(j);
						}
					}
				}
			}
		}
		else {
			for (int j = 0; j < layer->number_maps; j++) {
				for (auto k = channel_connection->equal_range(j).first; k != channel_connection->equal_range(j).second; k++) {
					this->channel_connection[0][j].push_back(k->second);
					this->channel_connection[1][k->second].push_back(j);
				}
			}
		}

		// add last index (-1)
		for (int j = 0; j < layer->number_maps; j++) {
			this->channel_connection[0][j].push_back(-1);
		}
		for (int j = 0; j < parent_layer->number_maps; j++) {
			this->channel_connection[1][j].push_back(-1);
		}

		{ // copy channel_connection[0]
			int max = 0, *memory;

			for (int j = 0; j < layer->number_maps; j++) {
				if (max < this->channel_connection[0][j].size()) {
					max = static_cast<int>(this->channel_connection[0][j].size());
				}
			}
			cudaMalloc(&channel_connection_device[0], sizeof(int) * max * layer->number_maps);
			memory = new int[max * layer->number_maps];

			for (int j = 0; j < layer->number_maps; j++) {
				for (int i = 0; i < this->channel_connection[0][j].size(); i++) {
					memory[j + i * layer->number_maps] = this->channel_connection[0][j][i];
				}
			}
			cudaMemcpy(channel_connection_device[0], memory, sizeof(int) * max * layer->number_maps, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		{ // copy channel_connection[1]
			int max = 0, *memory;

			for (int j = 0; j < parent_layer->number_maps; j++) {
				if (max < this->channel_connection[1][j].size()) {
					max = static_cast<int>(this->channel_connection[1][j].size());
				}
			}
			cudaMalloc(&channel_connection_device[1], sizeof(int) * max * parent_layer->number_maps);
			memory = new int[max * parent_layer->number_maps];

			for (int j = 0; j < parent_layer->number_maps; j++) {
				for (int i = 0; i < this->channel_connection[1][j].size(); i++) {
					memory[j + i * parent_layer->number_maps] = this->channel_connection[1][j][i];
				}
			}
			cudaMemcpy(channel_connection_device[1], memory, sizeof(int) * max * parent_layer->number_maps, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		// remove last index (-1)
		for (int j = 0; j < layer->number_maps; j++) {
			this->channel_connection[0][j].pop_back();
		}
		for (int j = 0; j < parent_layer->number_maps; j++) {
			this->channel_connection[1][j].pop_back();
		}
	}

	{ // time_connection
		if (time_connection == nullptr) {
			if (!strstr(properties.c_str(), "copy")) {
				for (int t = 0; t < layer->time_step; t++) {
					for (int u = 0; u < parent_layer->time_step; u++) {
						if ((parent_layer->time_mask == nullptr && t == u) || (parent_layer->time_mask && parent_layer->time_mask[u])) {
							this->time_connection[0][t].push_back(u);
							this->time_connection[1][u].push_back(t);
						}
					}
				}
			}
			else if (parent_layer->time_step * parent_layer->number_nodes != layer->time_step * layer->number_nodes) {
				cerr << "[Connection], copy connection but parent layer(" << parent_layer->time_step << " x " << parent_layer->number_nodes << ") != layer(" << layer->time_step << " x " << layer->number_nodes << ")" << endl;
			}
		}
		else {
			for (int t = 0; t < layer->time_step; t++) {
				for (auto u = time_connection->equal_range(t).first; u != time_connection->equal_range(t).second; u++) {
					this->time_connection[0][t].push_back(u->second);
					this->time_connection[1][u->second].push_back(t);
				}
			}
		}

		// add last index (-1)
		for (int t = 0; t < layer->time_step; t++) {
			this->time_connection[0][t].push_back(-1);
		}
		for (int t = 0; t < parent_layer->time_step; t++) {
			this->time_connection[1][t].push_back(-1);
		}

		{ // copy time_connection[0]
			int max = 0, *memory;

			for (int t = 0; t < layer->time_step; t++) {
				if (max < this->time_connection[0][t].size()) {
					max = static_cast<int>(this->time_connection[0][t].size());
				}
			}
			cudaMalloc(&time_connection_device[0], sizeof(int) * max * layer->time_step);
			memory = new int[max * layer->time_step];

			for (int t = 0; t < layer->time_step; t++) {
				for (int i = 0; i < this->time_connection[0][t].size(); i++) {
					memory[t + i * layer->time_step] = this->time_connection[0][t][i];
				}
			}
			cudaMemcpy(time_connection_device[0], memory, sizeof(int) * max * layer->time_step, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		{ // copy time_connection[1]
			int max = 0, *memory;

			for (int t = 0; t < parent_layer->time_step; t++) {
				if (max < this->time_connection[1][t].size()) {
					max = static_cast<int>(this->time_connection[1][t].size());
				}
			}
			cudaMalloc(&time_connection_device[1], sizeof(int) * max * parent_layer->time_step);
			memory = new int[max * parent_layer->time_step];

			for (int t = 0; t < parent_layer->time_step; t++) {
				for (int i = 0; i < this->time_connection[1][t].size(); i++) {
					memory[t + i * parent_layer->time_step] = this->time_connection[1][t][i];
				}
			}
			cudaMemcpy(time_connection_device[1], memory, sizeof(int) * max * parent_layer->time_step, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		// remove last index (-1)
		for (int t = 0; t < layer->time_step; t++) {
			this->time_connection[0][t].pop_back();
		}
		for (int t = 0; t < parent_layer->time_step; t++) {
			this->time_connection[1][t].pop_back();
		}
	}

	// allocate memory for the weight, if necessary
	if (properties[0] == 'W') {
		for (int j = 0, index = 0; j < layer->number_maps; j++) {
			for (auto k = this->channel_connection[0][j].begin(); k != this->channel_connection[0][j].end(); k++) {
				for (int l = 0; l < kernel_size; l++) {
					weight_index.insert(pair<int, int>(j * parent_layer->number_maps * kernel_size + (*k) * kernel_size + l, index++));
				}
				number_weights += kernel_size;
			}
			if (j == 0) {
				number_weights_per_map = number_weights;
			}
		}
		cudaMalloc(&weight, sizeof(float) * number_weights);
		cudaMemset(weight, 0, sizeof(float) * number_weights);
	}

	if (properties[0] == 'P' || properties[0] == 'W') {
		int offset[3] = { kernel_depth - (abs(layer->map_depth * stride_depth - parent_layer->map_depth) + 1), kernel_height - (abs(layer->map_height * stride_height - parent_layer->map_height) + 1), kernel_width - (abs(layer->map_width * stride_width - parent_layer->map_width) + 1) };

		from_error = new vector<Index>[parent_layer->map_size];
		from_neuron = new vector<Index>[layer->map_size];
		from_weight = (number_weights) ? (new vector<Index>[kernel_size]) : (nullptr);

		for (int k = 0; k < layer->map_depth; k++) {
			for (int l = 0; l < layer->map_height; l++) {
				for (int m = 0; m < layer->map_width; m++) {
					int node_index[2] = { k * layer->map_height * layer->map_width + l * layer->map_width + m, };

					if (properties[0] == 'W') {
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

												node_index[1] = o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

												index.prev_node = node_index[1];
												index.next_node = node_index[0];
												index.weight = weight_index.find((distance[0] + offset[0]) * kernel_height * kernel_width + (distance[1] + offset[1]) * kernel_width + (distance[2] + offset[2]))->second;

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

												node_index[1] = o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

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

		{ // add last index (index.weight = -1)
			Index index;

			index.weight = -1;

			for (int j = 0; j < parent_layer->map_size; j++) {
				for (auto index = from_error[j].begin(); index != from_error[j].end(); index++) {
					if (index->weight == -1) {
						index->weight = from_error[j].size();
					}
				}
				from_error[j].push_back(index);
			}
			for (int j = 0; j < layer->map_size; j++) {
				for (auto index = from_neuron[j].begin(); index != from_neuron[j].end(); index++) {
					if (index->weight == -1) {
						index->weight = from_neuron[j].size();
					}
				}
				from_neuron[j].push_back(index);
			}
			if (from_weight) {
				for (int j = 0; j < this->kernel_size; j++) {
					from_weight[j].push_back(index);
				}
			}
		}

		{ // copy from_error
			int max = 0;

			Index *memory = new Index[0];

			for (int j = 0; j < parent_layer->map_size; j++) {
				if (max < from_error[j].size()) {
					max = static_cast<int>(from_error[j].size());
				}
			}
			cudaMalloc(&this->from_error_device, sizeof(Index) * max * parent_layer->map_size);
			memory = new Index[max * parent_layer->map_size];

			for (int j = 0; j < parent_layer->map_size; j++) {
				for (int k = 0; k < from_error[j].size(); k++) {
					memory[j + k * parent_layer->map_size] = from_error[j][k];
				}
			}
			cudaMemcpy(this->from_error_device, memory, sizeof(Index) * max * parent_layer->map_size, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		{ // copry from_neuron
			int max = 0;

			Index *memory;

			for (int j = 0; j < layer->map_size; j++) {
				if (max < from_neuron[j].size()) {
					max = static_cast<int>(from_neuron[j].size());
				}
			}
			cudaMallocManaged(&this->from_neuron_device, sizeof(Index) * max * layer->map_size);
			memory = new Index[max * layer->map_size];

			for (int j = 0; j < layer->map_size; j++) {
				for (int k = 0; k < from_neuron[j].size(); k++) {
					memory[j + k * layer->map_size] = from_neuron[j][k];
				}
			}
			cudaMemcpy(this->from_neuron_device, memory, sizeof(Index) * max * layer->map_size, cudaMemcpyHostToDevice);
			delete[] memory;
		}

		if (from_weight) {
			int max = 0;

			Index *memory;

			for (int j = 0; j < kernel_size; j++) {
				if (max < from_weight[j].size()) {
					max = static_cast<int>(from_weight[j].size());
				}
			}
			cudaMallocManaged(&this->from_weight_device, sizeof(Index) * max * kernel_size);
			memory = new Index[max * kernel_size];

			for (int j = 0; j < kernel_size; j++) {
				for (int k = 0; k < from_weight[j].size(); k++) {
					memory[j + k * kernel_size] = from_weight[j][k];
				}
			}
			cudaMemcpy(this->from_weight_device, memory, sizeof(Index) * max * kernel_size, cudaMemcpyHostToDevice);
			delete[] memory;
		}
		delete[] from_error;
		delete[] from_neuron;
		delete[] from_weight;
	}
}
Connection::~Connection() {}

void Connection::Destruct() {
	if (from_error_device) {
		cudaFree(from_error_device);
	}
	if (from_neuron_device) {
		cudaFree(from_neuron_device);
	}
	if (from_weight_device) {
		cudaFree(from_weight_device);
	}
	if (initializer) {
		delete initializer;
	}
	if (optimizer) {
		optimizer->Destruct();
		delete optimizer;
	}
	cudaFree(channel_connection_device[0]);
	cudaFree(channel_connection_device[1]);
	cudaFree(time_connection_device[0]);
	cudaFree(time_connection_device[1]);
	cudaFree(weight);
	delete[] channel_connection[0];
	delete[] channel_connection[1];
	delete[] time_connection[0];
	delete[] time_connection[1];
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
			number_maps[1] = layer->number_maps;
		}
		if (strstr(properties.c_str(), "pointwise") && parent_layer->Search_Connection("depthwise")) {
			kernel_size = parent_layer->Search_Connection("depthwise")->kernel_size;
			parent_layer = parent_layer->Search_Connection("depthwise")->parent_layer;
			number_maps[0] = parent_layer->number_maps;
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

Connection *Connection::Copy(int type) {
	Connection *connection;

	unordered_multimap<int, int> channel_connection;
	unordered_multimap<int, int> time_connection;

	for (int j = 0; j < layer->number_maps; j++) {
		for (auto k = this->channel_connection[0][j].begin(); k != this->channel_connection[0][j].end(); k++) {
			channel_connection.insert(pair<int, int>(j, *k));
		}
	}
	for (int t = 0; t < layer->time_step; t++) {
		for (auto u = this->time_connection[0][t].begin(); u != this->time_connection[0][t].end(); u++) {
			time_connection.insert(pair<int, int>(t, *u));
		}
	}
	connection = new Connection(layer, parent_layer, properties, &channel_connection, &time_connection, type);
	connection->initializer = initializer->Copy();
	return connection;
}
Connection *Connection::Initializer(::Initializer initializer) {
	if (this->initializer) {
		delete this->initializer;
	}
	this->initializer = initializer.Copy();
	return this;
}


CTC::CTC(int number_labels, string label[], string blank) {
	this->number_labels = number_labels;
	this->label = new string[this->number_labels];

	for (int i = 0; i < number_labels; i++) {
		label_index.insert(pair<string, int>(this->label[i] = label[i], i));
	}
	for (int i = 0; i < number_labels; i++) {
		if (label[i] == blank) {
			break;
		}
		if (i == number_labels - 1) {
			cerr << "[CTC], label does not contain blank(" << blank << ")." << endl;
		}
	}
}
CTC::~CTC() {
	if (label) {
		delete[] label;
	}
}

void CTC::Decode(vector<string> &hypothesis, int sequence_length, float _likelihood[]) {
	hypothesis.clear();

	for (int t = 0, argmax, previous_state = number_labels - 1; t < sequence_length; t++) {
		float max;

		float *likelihood = &_likelihood[t * number_labels];

		for (int i = 0; i < number_labels; i++) {
			if (i == 0 || max < likelihood[i]) {
				max = likelihood[argmax = i];
			}
		}
		if (previous_state != argmax) {
			if (label[argmax] != blank) {
				hypothesis.push_back(label[argmax]);
			}
			previous_state = argmax;
		}
	}
}

int CTC::Search_Label(string label) {
	auto l = label_index.find(label);

	if (l != label_index.end()) {
		return l->second;
	}
	cerr << "[Search_Label], label '" + label + "' not found." << endl;
	return 0;
}

double CTC::Backward_Algorithm(vector<string> &reference, int sequence_length, float _likelihood[], double **beta) {
	double log_likelihood = 0;

	for (int t = sequence_length - 1, length_reference = static_cast<int>(reference.size()); t >= 0; t--) {
		float *likelihood = &_likelihood[t * number_labels];

		double sum = -numeric_limits<double>::infinity();

		if (t == sequence_length - 1) {
			for (int s = 0; s < length_reference; s++) {
				beta[t][s] = (s >= length_reference - 2) ? (log(likelihood[Search_Label(reference[s])])) : (-numeric_limits<double>::infinity());
			}
		}
		else {
			for (int s = 0; s < length_reference; s++) {
				double sum = -numeric_limits<double>::infinity();

				if (s <= 2 * t + 1) {
					if (reference[s] == blank || (s <= length_reference - 3 && reference[s + 2] == reference[s])) {
						sum = (s == length_reference - 1) ? (beta[t + 1][s]) : (Log_Add(beta[t + 1][s], beta[t + 1][s + 1]));
					}
					else {
						sum = (s == length_reference - 2) ? (Log_Add(beta[t + 1][s], beta[t + 1][s + 1])) : (Log_Add(Log_Add(beta[t + 1][s], beta[t + 1][s + 1]), beta[t + 1][s + 2]));
					}
				}
				beta[t][s] = sum + log(likelihood[Search_Label(reference[s])]);
			}
		}
		for (int s = 0; s < length_reference; s++) {
			sum = Log_Add(sum, beta[t][s]);
		}
		for (int s = 0; s < length_reference; s++) {
			beta[t][s] -= sum;
		}
		log_likelihood += sum;
	}
	return log_likelihood;
}
double CTC::Calculate_Error(vector<string> &reference, int sequence_length, float *error, float *likelihood) {
	double log_likelihood;

	double **alpha = new double*[sequence_length];
	double **beta = new double*[sequence_length];

	for (int t = 0; t < sequence_length; t++) {
		alpha[t] = new double[reference.size()];
		beta[t] = new double[reference.size()];
	}
	log_likelihood = Forward_Algorithm(reference, sequence_length, likelihood, alpha);
	Backward_Algorithm(reference, sequence_length, likelihood, beta);

	for (int t = 0; t < sequence_length; t++) {
		int index = t * number_labels;

		double sum[] = { -numeric_limits<double>::infinity(), };

		for (int j = 0; j < reference.size(); j++) {
			sum[0] = Log_Add(sum[0], alpha[t][j] + beta[t][j] - log(likelihood[index + Search_Label(reference[j])]));
		}
		for (int i = 0; i < number_labels; i++) {
			sum[1] = -numeric_limits<double>::infinity();

			for (int j = 0; j < reference.size(); j++) {
				if (i == Search_Label(reference[j])) {
					sum[1] = Log_Add(sum[1], alpha[t][j] + beta[t][j]);
				}
			}
			error[index + i] = likelihood[index + i] - exp(sum[1] - log(likelihood[index + i]) - sum[0]);

			if (!isfinite(error[index + i])) {
				error[index + i] = 0;
			}
		}
	}

	for (int t = 0; t < sequence_length; t++) {
		delete[] alpha[t];
		delete[] beta[t];
	}
	delete[] alpha;
	delete[] beta;

	return log_likelihood;
}
double CTC::Forward_Algorithm(vector<string> &reference, int sequence_length, float _likelihood[], double **alpha) {
	double log_likelihood = 0;

	for (int t = 0, length_reference = static_cast<int>(reference.size()); t < sequence_length; t++) {
		float *likelihood = &_likelihood[t * number_labels];

		double sum = -numeric_limits<double>::infinity();

		if (t == 0) {
			for (int s = 0; s < length_reference; s++) {
				alpha[t][s] = (s <= 1) ? (log(likelihood[Search_Label(reference[s])])) : (-numeric_limits<double>::infinity());
			}
		}
		else {
			for (int s = 0; s < length_reference; s++) {
				double sum = -numeric_limits<double>::infinity();

				if (s >= (length_reference - 1) - 2 * ((sequence_length - 1) - t) - 1) {
					if (reference[s] == blank || (s >= 2 && reference[s - 2] == reference[s])) {
						sum = (s == 0) ? (alpha[t - 1][s]) : (Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1]));
					}
					else {
						sum = (s == 1) ? (Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1])) : (Log_Add(Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1]), alpha[t - 1][s - 2]));
					}
				}
				alpha[t][s] = sum + log(likelihood[Search_Label(reference[s])]);
			}
		}
		for (int s = 0; s < length_reference; s++) {
			sum = Log_Add(sum, alpha[t][s]);
		}
		for (int s = 0; s < length_reference; s++) {
			alpha[t][s] -= sum;
		}
		log_likelihood += sum;
	}
	return log_likelihood;
}
double CTC::Log_Add(double a, double b) {
	double max;

	if (!isfinite(a) && !isfinite(b)) {
		return -numeric_limits<double>::infinity();
	}
	if (!isfinite(a)) {
		return b;
	}
	if (!isfinite(b)) {
		return a;
	}
	max = (a > b) ? (a) : (b);
	return (max + log1p(exp(a + b - 2 * max)));
}

CTC* CTC::Copy() {
	return new CTC(number_labels, label, blank);
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
Initializer::Initializer(Constant initializer) {
	generator = nullptr;
	this->value = initializer.value;
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
Initializer::Matrix::Matrix(int number_rows, int number_columns) {
	this->number_columns = number_columns;
	this->number_rows = number_rows;

	memset(data = new double[number_rows * number_columns], 0, sizeof(double) * number_rows * number_columns);
}
Initializer::Matrix::~Matrix() {
	delete[] data;
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
void Initializer::Matrix::Gram_Schmidt_Process(double gain) {
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
void Initializer::Matrix::Identity() {
	for (int i = 0; i < number_rows; i++) {
		for (int j = 0; j < number_columns; j++) {
			(*this)(i, j) = (i == j);
		}
	}
}
void Initializer::Matrix::LQ_Decomposition(Matrix &L, Matrix &Q) {
	Matrix A = (*this);

	A.Transpose();
	A.QR_Decomposition(Q, L);
	Q.Transpose();
	L.Transpose();
}
void Initializer::Matrix::QR_Decomposition(Matrix &Q, Matrix &R) {
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
void Initializer::Matrix::Transpose() {
	Matrix T(number_columns, number_rows);

	for (int i = 0; i < number_rows; i++) {
		for (int j = 0; j < number_columns; j++) {
			T(j, i) = (*this)(i, j);
		}
	}
	(*this) = T;
}

Initializer::Matrix Initializer::Matrix::Multiplication(const Matrix &A, const Matrix &B) {
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
	this->time_step = 1;

	Construct();
}
Layer::Layer(int time_step, int number_maps, string properties) {
	this->map_width = 1;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, string properties) {
	this->map_width = map_width;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, int map_height, string properties) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct();
}
Layer::Layer(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct();
}
Layer::~Layer() {}

void Layer::Activate(int time_index, bool training) {
	if (lstm) {
		lstm->Activate(time_index, training);
	}
	else if (rnn) {
		rnn->Activate(time_index, training);
	}
	else {
		dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

		if (bias) {
			::Add_Bias << <number_blocks, NUMBER_THREADS >> > (*this, time_index);
		}
		if (batch_normalization) {
			batch_normalization->Activate(time_index, neuron, training);
		}
		if (dropout) {
			if (training && time_index == 0) {
				dropout->Initialize_Mask();
			}
			::Activate << <number_blocks, NUMBER_THREADS >> > (*this, time_index, *dropout, training);
		}
		else {
			::Activate << <number_blocks, NUMBER_THREADS >> > (*this, time_index);
		}
	}
}
void Layer::Adjust_Parameter(int iterations) {
	if (lstm) {
		lstm->Adjust_Parameter(iterations);
	}
	else if (rnn) {
		rnn->Adjust_Parameter(iterations);
	}
	else {
		if (batch_normalization) {
			batch_normalization->Adjust_Parameter(iterations);
		}

		// adjust bias
		if (bias) {
			::Adjust_Bias << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer, iterations);
		}

		// adjust weight
		for (int i = 0; i < connection.size(); i++) {
			Connection *connection = this->connection[i];

			if (connection->properties[0] == 'W') {
				::Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*connection->layer, *connection->parent_layer, *connection, *connection->optimizer, iterations);
			}
		}
	}
}
void Layer::Backward(int time_index) {
	if (lstm) {
		lstm->Backward(time_index);
	}
	else if (rnn) {
		rnn->Backward(time_index);
	}
	else {
		int t = time_index;

		for (int k = 0; k < connection.size(); k++) {
			Connection *connection = this->connection[k];

			Layer *parent_layer = connection->parent_layer;

			dim3 number_blocks(parent_layer->batch_size, parent_layer->number_nodes / NUMBER_THREADS + 1);

			if (connection->properties[0] == 'P'){
				if (strstr(connection->properties.c_str(), "average")) {
					::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 0);
				}
				else if (strstr(connection->properties.c_str(), "max")) {
					::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 1);
				}
			}
			else if (connection->properties[0] == 'W') {
				::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 2);
			}
			else if (strstr(connection->properties.c_str(), "copy")) {
				::Copy_Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
			}
			if (strstr(connection->properties.c_str(), "copy") && t == time_step - 1) {
				for (int t = 0; t < time_step; t++) {
					if (connection->time_connection[1][t].size() > 0) {
						return;
					}
				}
				cudaMemcpy(parent_layer->error, error, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
			}
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
	if (lstm) {
		lstm->Compile(optimizer);
	}
	if (rnn) {
		rnn->Compile(optimizer);
	}
	Optimizer(optimizer);
	Initialize();
}
void Layer::Construct() {
	this->activation = Activation::linear;
	this->batch_normalization = nullptr;
	this->batch_size = 1;
	this->initializer = nullptr;
	this->lstm = nullptr;
	this->map_size = map_depth * map_height * map_width;
	this->number_nodes = number_maps * map_size;
	this->dropout = (strstr(properties.c_str(), "dropout")) ? (new Dropout(number_nodes, atof(strstr(properties.c_str(), "dropout") + 7))) : (nullptr);
	this->optimizer = nullptr;
	this->rnn = nullptr;
	this->time_mask = nullptr;

	bias = nullptr;
	cudaMalloc(&error, sizeof(float) * time_step * number_nodes);
	cudaMalloc(&neuron, sizeof(float) * time_step * number_nodes);
}
void Layer::Differentiate(int time_index, Loss *loss, float **y_batch) {
	if (lstm) {
		lstm->Differentiate(time_index);
	}
	else if (rnn) {
		rnn->Differentiate(time_index);
	}
	else {
		int t = time_index;

		dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

		if (y_batch) {
			if (time_mask == nullptr || time_mask[t]) {
				float *y_data, *memory = new float[batch_size * time_step * number_nodes];

				for (int h = 0; h < batch_size; h++) {
					memcpy(&memory[h * time_step * number_nodes], y_batch[h], sizeof(float) * time_step * number_nodes);
				}
				cudaMalloc(&y_data, sizeof(float) * batch_size * time_step * number_nodes);
				cudaMemcpy(y_data, memory, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyHostToDevice);

				::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, time_index, loss->type, y_data);

				cudaFree(y_data);
				delete[] memory;

				Differentiate(time_index, loss);
			}
		}
		else {
			if (dropout) {
				::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, *dropout, time_index);
			}
			else {
				::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, time_index);
			}
			if (batch_normalization) {
				batch_normalization->Differentiate(time_index, error);
			}
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
	if (lstm) {
		lstm->Destruct();
		delete lstm;
	}
	if (optimizer) {
		optimizer->Destruct();
		delete optimizer;
	}
	if (rnn) {
		rnn->Destruct();
		delete rnn;
	}
	for (int i = 0; i < connection.size(); i++) {
		connection[i]->Destruct();
		delete connection[i];
	}
	cudaFree(error);
	cudaFree(neuron);
}
void Layer::Forward(int time_index) {
	if (lstm) {
		lstm->Forward(time_index);
	}
	else if (rnn) {
		rnn->Forward(time_index);
	}
	else {
		int t = time_index;

		dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

		for (int k = 0; k < connection.size(); k++) {
			Connection *connection = this->connection[k];

			Layer *parent_layer = connection->parent_layer;

			if (connection->properties[0] == 'P') {
				if (strstr(connection->properties.c_str(), "average")) {
					::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 0);
				}
				else if (strstr(connection->properties.c_str(), "max")) {
					::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 1);
				}
			}
			else if (connection->properties[0] == 'W') {
				::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t, 2);
			}
			if (strstr(connection->properties.c_str(), "copy")) {
				::Copy_Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
			}
			if (strstr(connection->properties.c_str(), "copy") && t == 0) {
				for (int t = 0; t < time_step; t++) {
					if (connection->time_connection[0][t].size() > 0) {
						return;
					}
				}
				cudaMemcpy(neuron, parent_layer->neuron, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
			}
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
		this->optimizer = optimizer->Copy(number_maps);
	}
}
void Layer::Resize_Memory(int batch_size) {
	int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

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
	if (lstm) {
		lstm->Resize_Memory(batch_size);
	}
	if (rnn) {
		rnn->Resize_Memory(batch_size);
	}
	cudaMemset(error, 0, memory_size);
	cudaMemset(neuron, 0, memory_size);
}

Batch_Normalization* Layer::Batch_Normalization(double epsilon, double momentum) {
	if (batch_normalization) {
		delete batch_normalization;
	}
	return (batch_normalization = new ::Batch_Normalization(time_step, number_maps, map_size, epsilon, momentum, this));
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
	return new Layer(time_step, number_maps, map_width, map_height, map_depth, properties);
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
Layer* Layer::Time_Mask(bool time_mask[], int length_mask) {
	if (length_mask == 0) {
		length_mask = time_step;
	}
	if (this->time_mask) {
		delete[] this->time_mask;
	}
	memset(this->time_mask = new bool[time_step], 0, time_step);
	memcpy(this->time_mask, time_mask, length_mask);
	return this;
}


LSTM::LSTM(int time_step, int number_nodes, string properties, Layer *layer) {
	this->map_width = 1;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_nodes;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
LSTM::LSTM(int time_step, int number_maps, int map_width, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
LSTM::LSTM(int time_step, int number_maps, int map_width, int map_height, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
LSTM::LSTM(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
LSTM::~LSTM() {}

void LSTM::Activate(int time_index, bool training) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	for (int i = 0; i < number_node_types - 1; i++) {
		if (batch_normalization[i][0]) {
			batch_normalization[i][0]->Activate(t, neuron[i][0], training);
		}
		if (batch_normalization[i][1]) {
			batch_normalization[i][1]->Activate(t, neuron[i][1], training);
		}
	}
	::Activate << <number_blocks, NUMBER_THREADS >> > (*this, t);

	if (batch_normalization[cell_output][0]) {
		batch_normalization[cell_output][0]->Activate(t, neuron[cell_output][0], training);
	}
	::Activate << <number_blocks, NUMBER_THREADS >> > (*this, *layer, t);
}
void LSTM::Adjust_Parameter(int iterations) {
	for (int i = 0; i < number_node_types; i++) {
		for (int j = 0; j < 2; j++) {
			if (batch_normalization[i][j]) {
				batch_normalization[i][j]->Adjust_Parameter(iterations);
			}
		}
	}

	// adjust bias
	for (int i = 0; i < number_weight_types; i++) {
		if (batch_normalization[i][0]) {
			::Adjust_Bias << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer[i], *batch_normalization[i][0], i, iterations);
		}
		else {
			::Adjust_Bias << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer[i], i, iterations);
		}
	}

	// adjust weight
	for (int i = 0; i < layer->connection.size(); i++) {
		Connection *connection = layer->connection[i];

		if (connection->properties[0] == 'W'){
			if (strstr(connection->properties.c_str(), "recurrent")) {
				Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *layer, *connection, *connection->optimizer, iterations, true);
			}
			else {
				Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, *connection->optimizer, iterations);
			}
		}
	}
}
void LSTM::Backward(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	for (int k = 0; k < layer->connection.size(); k++) {
		Connection *connection = layer->connection[k];

		Layer *parent_layer = connection->parent_layer;

		if (connection->properties[0] == 'W' && strstr(connection->properties.c_str(), "recurrent")) {
			if ((direction == 1 && t > 0) || (direction == -1 && t < time_step - 1)) {
				dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

				::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *layer, *connection, t, true);
			}
		}
		if (connection->properties[0] == 'W' && !strstr(connection->properties.c_str(), "recurrent")) {
			dim3 number_blocks(parent_layer->batch_size, parent_layer->number_nodes / NUMBER_THREADS + 1);

			::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		else if (strstr(connection->properties.c_str(), "copy")) {
			dim3 number_blocks(parent_layer->batch_size, parent_layer->number_nodes / NUMBER_THREADS + 1);

			::Copy_Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		if (strstr(connection->properties.c_str(), "copy") && time_index == time_step - 1) {
			for (int t = 0; t < time_step; t++) {
				if (connection->time_connection[1][t].size() > 0) {
					return;
				}
			}
			cudaMemcpy(parent_layer->error, this->error[connection->type][0], sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
		}
	}
}
void LSTM::Compile(::Optimizer *optimizer) {
	if (batch_normalization[0][0]) {
		for (int i = 0; i < number_node_types; i++) {
			if (i > 0) {
				if (batch_normalization[i][0]) {
					delete batch_normalization[i][0];
				}
				batch_normalization[i][0] = batch_normalization[0][0]->Copy();
			}
			if (i < number_node_types - 1) {
				if (batch_normalization[i][1]) {
					delete batch_normalization[i][1];
				}
				batch_normalization[i][1] = batch_normalization[0][0]->Copy();
			}
		}
	}
	Optimizer(optimizer);
	Initialize();
}
void LSTM::Construct(Layer *layer) {
	if (layer) {
		activation = Activation::tanh;
		batch_size = 1;
		direction = 1;
		map_size = map_depth * map_height * map_width;
		number_nodes = number_maps * map_size;
		recurrent_activation = Activation::sigmoid;
		time_mask = nullptr;

		for (int i = 0; i < number_node_types; i++) {
			for (int j = 0; j < 2; j++) {
				batch_normalization[i][j] = nullptr;
				cudaMalloc(&error[i][j], sizeof(float) * time_step * number_nodes);
				cudaMalloc(&neuron[i][j], sizeof(float) * time_step * number_nodes);
			}
		}
		for (int i = 0; i < number_weight_types; i++) {
			cudaMalloc(&bias[i], sizeof(float) * number_maps);
			initializer[i] = new ::Initializer(0);
			optimizer[i] = nullptr;
		}
	}
}
void LSTM::Differentiate(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	::DIfferentiate << <number_blocks, NUMBER_THREADS >> > (*this, *layer, t, 0);

	if (batch_normalization[cell_output][0]) {
		batch_normalization[cell_output][0]->Differentiate(t, error[cell_output][0]);
	}
	::DIfferentiate << <number_blocks, NUMBER_THREADS >> > (*this, *layer, t, 1);

	for (int i = 0; i < number_node_types - 1; i++) {
		if (batch_normalization[i][0]) {
			batch_normalization[i][0]->Differentiate(t, error[i][0]);
		}
		if (batch_normalization[i][1]) {
			batch_normalization[i][1]->Differentiate(t, error[i][1]);
		}
	}
}
void LSTM::Destruct() {
	if (layer) {
		for (int i = 0; i < number_node_types; i++) {
			if (batch_normalization[i][0]) {
				batch_normalization[i][0]->Destruct();
				delete batch_normalization[i][0];
			}
			if (batch_normalization[i][1]) {
				batch_normalization[i][1]->Destruct();
				delete batch_normalization[i][1];
			}
			for (int j = 0; j < 2; j++) {
				cudaFree(error[i][j]);
				cudaFree(neuron[i][j]);
			}
		}
		for (int i = 0; i < number_weight_types; i++) {
			if (bias[i]) {
				cudaFree(bias[i]);
			}
			if (initializer[i]) {
				delete initializer[i];
			}
			if (optimizer[i]) {
				optimizer[i]->Destruct();
				delete optimizer[i];
			}
		}
		if (time_mask) {
			delete[] time_mask;
		}
	}
}
void LSTM::Forward(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	for (int k = 0; k < layer->connection.size(); k++) {
		Connection *connection = layer->connection[k];

		Layer *parent_layer = connection->parent_layer;

		if (connection->properties[0] == 'W' && strstr(connection->properties.c_str(), "recurrent")) {
			if ((direction == 1 && t > 0) || (direction == -1 && t < time_step - 1)) {
				::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *layer, *connection, t, true);
			}
		}
		if (connection->properties[0] == 'W' && !strstr(connection->properties.c_str(), "recurrent")) {
			::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		else if (strstr(connection->properties.c_str(), "copy")) {
			::Copy_Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		if (strstr(connection->properties.c_str(), "copy") && time_index == 0) {
			for (int t = 0; t < time_step; t++) {
				if (connection->time_connection[0][t].size() > 0) {
					return;
				}
			}
			cudaMemcpy(neuron[connection->type][0], parent_layer->neuron, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
		}
	}
}
void LSTM::Initialize() {
	for (int i = 0; i < number_node_types; i++) {
		if (batch_normalization[i][0]) {
			batch_normalization[i][0]->Initialize();
		}
		if (batch_normalization[i][1]) {
			batch_normalization[i][1]->Initialize();
		}
	}
	for (int i = 0; i < number_weight_types; i++) {
		if (bias[i]) {
			initializer[i]->Random(number_maps, bias[i], 1, number_maps);
		}
	}
}
void LSTM::Optimizer(::Optimizer *optimizer) {
	for (int i = 0; i < number_node_types; i++) {
		if (batch_normalization[i][0]) {
			batch_normalization[i][0]->Optimizer(*optimizer);
		}
		if (batch_normalization[i][1]) {
			batch_normalization[i][1]->Optimizer(*optimizer);
		}
	}
	for (int i = 0; i < number_weight_types; i++) {
		if (bias[i]) {
			if (this->optimizer[i]) {
				this->optimizer[i]->Destruct();
				delete this->optimizer[i];
			}
			this->optimizer[i] = optimizer->Copy(number_maps);
		}
	}
}
void LSTM::Resize_Memory(int batch_size) {
	int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

	if (this->batch_size != batch_size) {
		for (int i = 0; i < number_node_types; i++) {
			for (int j = 0; j < 2; j++) {
				cudaFree(error[i][j]);
				cudaFree(neuron[i][j]);
				cudaMalloc(&error[i][j], memory_size);
				cudaMalloc(&neuron[i][j], memory_size);
			}
		}
		this->batch_size = batch_size;
	}
	for (int i = 0; i < number_node_types; i++) {
		for (int j = 0; j < 2; j++) {
			if (batch_normalization[i][j]) {
				batch_normalization[i][j]->Resize_Memory(batch_size);
			}
			cudaMemset(error[i][j], 0, memory_size);
			cudaMemset(neuron[i][j], 0, memory_size);
		}
	}
}

Batch_Normalization* LSTM::Batch_Normalization(double epsilon, double momentum) {
	if (batch_normalization[0][0]) {
		delete batch_normalization[0][0];
	}
	return (batch_normalization[0][0] = new ::Batch_Normalization(time_step, number_maps, map_size, epsilon, momentum, layer));
}

LSTM* LSTM::Activation(int activation) {
	this->activation = activation;
	return this;
}
LSTM* LSTM::Copy(Layer *layer) {
	return new LSTM(time_step, number_maps, map_width, map_height, map_depth, properties, (layer == nullptr) ? (this->layer) : (layer));
}
LSTM* LSTM::Direction(int direction) {
	if (direction != -1 && direction != 1) {
		cerr << "[Direction], {-1 (backwards), 1 (forwards)}" << endl;
		direction = 1;
	}
	this->direction = direction;
	return this;
}
LSTM* LSTM::Initializer(::Initializer initializer, int type) {
	for (int i = 0; i < number_weight_types; i++) {
		if (type == -1 || i == type) {
			if (this->initializer[i]) {
				delete this->initializer[i];
			}
			this->initializer[i] = initializer.Copy();
		}
	}
	return this;
}
LSTM* LSTM::Recurrent_Activation(int activation) {
	this->recurrent_activation = activation;
	return this;
}
LSTM* LSTM::Time_Mask(bool time_mask[], int length_mask) {
	if (length_mask == 0) {
		length_mask = time_step;
	}
	if (this->time_mask) {
		delete[] this->time_mask;
	}
	memset(this->time_mask = new bool[time_step], 0, time_step);
	memcpy(this->time_mask, time_mask, length_mask);
	return this;
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


RNN::RNN(int time_step, int number_nodes, string properties, Layer *layer) {
	this->map_width = 1;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_nodes;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
RNN::RNN(int time_step, int number_maps, int map_width, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = 1;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
RNN::RNN(int time_step, int number_maps, int map_width, int map_height, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = 1;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
RNN::RNN(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties, Layer *layer) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;
	this->time_step = time_step;

	Construct(this->layer = layer);
}
RNN::~RNN() {}

void RNN::Activate(int time_index, bool training) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	::Add_Bias << <number_blocks, NUMBER_THREADS >> > (*this, t);

	if (batch_normalization[0]) {
		batch_normalization[0]->Activate(t, neuron[0], training);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Activate(t, neuron[1], training);
	}
	::Activate << <number_blocks, NUMBER_THREADS >> > (*this, t);

	if (time_index == time_step - 1) {
		cudaMemcpy(layer->neuron, neuron[0], sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
	}
}
void RNN::Adjust_Parameter(int iterations) {
	if (batch_normalization[0]) {
		batch_normalization[0]->Adjust_Parameter(iterations);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Adjust_Parameter(iterations);
	}

	// adjust bias
	if (bias) {
		if (batch_normalization[0]) {
			::Adjust_Bias << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer, *batch_normalization[0], iterations);
		}
		else {
			::Adjust_Bias << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *optimizer, iterations);
		}
	}

	// adjust weight
	for (int i = 0; i < layer->connection.size(); i++) {
		Connection *connection = layer->connection[i];

		if (connection->properties[0] == 'W'){
			if (strstr(connection->properties.c_str(), "recurrent")) {
				::Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *connection, *connection->optimizer, iterations);
			}
			else {
				::Adjust_Weight << <connection->number_weights / NUMBER_THREADS + 1, NUMBER_THREADS >> > (*this, *connection->parent_layer, *connection, *connection->optimizer, iterations);
			}
		}
	}
}
void RNN::Backward(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	for (int k = 0; k < layer->connection.size(); k++) {
		Connection *connection = layer->connection[k];

		Layer *parent_layer = connection->parent_layer;

		if (connection->properties[0] == 'W' && strstr(connection->properties.c_str(), "recurrent")) {
			if ((direction == 1 && t > 0) || (direction == -1 && t < time_step - 1)) {
				dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

				::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *connection, t);
			}
		}
		if (connection->properties[0] == 'W' && !strstr(connection->properties.c_str(), "recurrent")) {
			dim3 number_blocks(parent_layer->batch_size, parent_layer->number_nodes / NUMBER_THREADS + 1);

			::Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		else if (strstr(connection->properties.c_str(), "copy")) {
			dim3 number_blocks(parent_layer->batch_size, parent_layer->number_nodes / NUMBER_THREADS + 1);

			::Copy_Backward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		if (strstr(connection->properties.c_str(), "copy") && time_index == time_step - 1) {
			for (int t = 0; t < time_step; t++) {
				if (connection->time_connection[1][t].size() > 0) {
					return;
				}
			}
			cudaMemcpy(parent_layer->error, error[0], sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
		}
	}
}
void RNN::Compile(::Optimizer *optimizer) {
	if (batch_normalization[0]) {
		if (batch_normalization[1]) {
			delete batch_normalization[1];
		}
		batch_normalization[1] = batch_normalization[0]->Copy();
	}
	Optimizer(optimizer);
	Initialize();
}
void RNN::Construct(Layer *layer) {
	if (layer) {
		activation = Activation::tanh;
		batch_normalization[0] = nullptr;
		batch_normalization[1] = nullptr;
		batch_size = 1;
		cudaMalloc(&bias, sizeof(float) * number_maps);
		direction = 1;
		initializer = new ::Initializer(0);
		map_size = map_depth * map_height * map_width;
		number_nodes = number_maps * map_size;
		optimizer = nullptr;
		time_mask = nullptr;

		for (int i = 0; i < 2; i++) {
			cudaMalloc(&error[i], sizeof(float) * time_step * number_nodes);
			cudaMalloc(&neuron[i], sizeof(float) * time_step * number_nodes);
		}
	}
}
void RNN::Destruct() {
	if (layer) {
		if (batch_normalization[0]) {
			delete batch_normalization[0];
		}
		if (batch_normalization[1]) {
			delete batch_normalization[1];
		}
		if (bias) {
			cudaFree(bias);
		}
		if (initializer) {
			delete initializer;
		}
		if (optimizer) {
			optimizer->Destruct();
			delete optimizer;
		}
		if (time_mask) {
			delete[] time_mask;
		}
		for (int i = 0; i < 2; i++) {
			cudaFree(error[i]);
			cudaFree(neuron[i]);
		}
	}
}
void RNN::Differentiate(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	if (time_index == time_step - 1) {
		cudaMemcpy(this->error[0], layer->error, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
	}
	::Differentiate << <number_blocks, NUMBER_THREADS >> > (*this, t);

	if (batch_normalization[0]) {
		batch_normalization[0]->Differentiate(t, error[0]);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Differentiate(t, error[1]);
	}
}
void RNN::Forward(int time_index) {
	int t = (direction == 1) ? (time_index) : (time_step - 1 - time_index);

	dim3 number_blocks(batch_size, number_nodes / NUMBER_THREADS + 1);

	for (int k = 0; k < layer->connection.size(); k++) {
		Connection *connection = layer->connection[k];

		Layer *parent_layer = connection->parent_layer;

		if (connection->properties[0] == 'W' && strstr(connection->properties.c_str(), "recurrent")) {
			if ((direction == 1 && t > 0) || (direction == -1 && t < time_step - 1)) {
				::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *connection, t);
			}
		}
		if (connection->properties[0] == 'W' && !strstr(connection->properties.c_str(), "recurrent")) {
			::Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		else if (strstr(connection->properties.c_str(), "copy")) {
			::Copy_Forward << <number_blocks, NUMBER_THREADS >> > (*this, *parent_layer, *connection, t);
		}
		if (strstr(connection->properties.c_str(), "copy") && time_index == 0) {
			for (int t = 0; t < time_step; t++) {
				if (connection->time_connection[0][t].size() > 0) {
					return;
				}
			}
			cudaMemcpy(neuron[0], parent_layer->neuron, sizeof(float) * batch_size * time_step * number_nodes, cudaMemcpyDeviceToDevice);
		}
	}
}
void RNN::Initialize() {
	if (batch_normalization[0]) {
		batch_normalization[0]->Initialize();
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Initialize();
	}
	if (bias) {
		initializer->Random(number_maps, bias, 1, number_maps);
	}
}
void RNN::Optimizer(::Optimizer *optimizer) {
	if (batch_normalization[0]) {
		batch_normalization[0]->Optimizer(*optimizer);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Optimizer(*optimizer);
	}
	if (bias) {
		if (this->optimizer) {
			this->optimizer->Destruct();
			delete this->optimizer;
		}
		this->optimizer = optimizer->Copy(number_maps);
	}
}
void RNN::Resize_Memory(int batch_size) {
	int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

	if (this->batch_size != batch_size) {
		for (int i = 0; i < 2; i++) {
			cudaFree(error[i]);
			cudaFree(neuron[i]);
			cudaMalloc(&error[i], memory_size);
			cudaMalloc(&neuron[i], memory_size);
		}
		this->batch_size = batch_size;
	}
	if (batch_normalization[0]) {
		batch_normalization[0]->Resize_Memory(batch_size);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Resize_Memory(batch_size);
	}
	for (int i = 0; i < 2; i++) {
		cudaMemset(error[i], 0, memory_size);
		cudaMemset(neuron[i], 0, memory_size);
	}
}

Batch_Normalization* RNN::Batch_Normalization(double epsilon, double momentum) {
	if (batch_normalization[0]) {
		delete batch_normalization[0];
	}
	return (batch_normalization[0] = new ::Batch_Normalization(time_step, number_maps, map_size, epsilon, momentum, layer));
}

RNN* RNN::Activation(int activation) {
	this->activation = activation;
	return this;
}
RNN* RNN::Copy(Layer *layer) {
	return new RNN(time_step, number_maps, map_width, map_height, map_depth, properties, (layer == nullptr) ? (this->layer) : (layer));
}
RNN* RNN::Direction(int direction) {
	if (direction != -1 && direction != 1) {
		cerr << "[Direction], {-1 (backwards), 1 (forwards)}" << endl;
		direction = 1;
	}
	this->direction = direction;
	return this;
}
RNN* RNN::Initializer(::Initializer initializer) {
	if (this->initializer) {
		delete this->initializer;
	}
	this->initializer = initializer.Copy();
	return this;
}
RNN* RNN::Time_Mask(bool time_mask[], int length_mask) {
	if (length_mask == 0) {
		length_mask = time_step;
	}
	if (this->time_mask) {
		delete[] this->time_mask;
	}
	memset(this->time_mask = new bool[time_step], 0, time_step);
	memcpy(this->time_mask, time_mask, length_mask);
	return this;
}


void Neural_Networks::Resize_Memory(int batch_size) {
	for (int i = 0; i < layer.size(); i++) {
		layer[i]->Resize_Memory(batch_size);
	}
	this->batch_size = batch_size;
}

double Neural_Networks::Calculate_Loss(Layer *layer, float **y_batch, vector<string> label[], int sequence_length_batch[], bool training) {
	float sum = 0;

	if (loss->type == Loss::connectionist_temporal_classification) {
		float *neuron = new float[batch_size * layer->time_step * layer->number_nodes];

		cudaMemcpy(neuron, layer->neuron, sizeof(float) * batch_size * layer->time_step * layer->number_nodes, cudaMemcpyDeviceToHost);

		if (training) {
			float *error = new float[batch_size * layer->time_step * layer->number_nodes];

			double *batch_sum = new double[batch_size];

			memset(error, 0, sizeof(float) * batch_size * layer->time_step * layer->number_nodes);

			#pragma omp parallel for
			for (int h = 0; h < batch_size; h++) {
				vector<string> reference;

				for (int j = 0; j < label[h].size(); j++) {
					reference.push_back(loss->ctc->blank);
					reference.push_back(label[h][j]);
				}
				reference.push_back(loss->ctc->blank);
				batch_sum[h] = loss->ctc->Calculate_Error(reference, (sequence_length_batch) ? (sequence_length_batch[h]) : (layer->time_step), &error[h * layer->time_step * layer->number_nodes], &neuron[h * layer->time_step * layer->number_nodes]);
			}
			for (int h = 0; h < batch_size; h++) {
				sum += batch_sum[h];
			}
			cudaMemcpy(layer->error, error, sizeof(float) * batch_size * layer->time_step * layer->number_nodes, cudaMemcpyHostToDevice);

			delete[] batch_sum;
			delete[] error;
		}
		else {
			#pragma omp parallel for
			for (int h = 0; h < batch_size; h++) {
				loss->ctc->Decode(label[h], (sequence_length_batch) ? (sequence_length_batch[h]) : (layer->time_step), &neuron[h * layer->time_step * layer->number_nodes]);
			}
		}
		delete[] neuron;
	}
	else {
		bool *time_mask = nullptr;

		float *y_data, *memory = new float[layer->batch_size * layer->time_step * layer->number_nodes];

		if (layer->time_mask) {
			cudaMalloc(&time_mask, layer->time_step);
			cudaMemcpy(time_mask, layer->time_mask, layer->time_step, cudaMemcpyHostToDevice);
		}
		cudaMalloc(&y_data, sizeof(float) * layer->batch_size * layer->time_step * layer->number_nodes);

		for (int h = 0; h < batch_size; h++) {
			memcpy(&memory[h * layer->time_step * layer->number_nodes], y_batch[h], sizeof(float) * layer->time_step * layer->number_nodes);
		}
		cudaMemcpy(y_data, memory, sizeof(float) * layer->batch_size * layer->time_step * layer->number_nodes, cudaMemcpyHostToDevice);
		::Calculate_Loss << <1, NUMBER_THREADS >> > (*layer, time_mask, this->loss->type, y_data);
		cudaMemcpy(&sum, y_data, sizeof(float), cudaMemcpyDeviceToHost);

		if (time_mask) {
			cudaFree(time_mask);
		}
		cudaFree(y_data);
		delete[] memory;
	}
	return sum;
}
double Neural_Networks::Evaluate(float **x_test, float **y_test, vector<string> hypothesis[], int sequence_length[], int test_size, int batch_size) {
	int *sequence_length_batch = (sequence_length) ? (new int[batch_size]) : (nullptr);
	
	float **x_batch = new float*[batch_size];
	float **y_batch = (y_test) ? (new float*[batch_size]) : (nullptr);

	double loss = 0;

	for (int g = 0, h = 0; g < test_size; g++) {
		if (x_batch) {
			x_batch[h] = x_test[g];
		}
		if (y_batch) {
			y_batch[h] = y_test[g];
		}
		if (sequence_length_batch) {
			sequence_length_batch[h] = sequence_length[g];
		}

		if (++h == batch_size || g == test_size - 1) {
			float *memory = new float[h * layer[0]->time_step * layer[0]->number_nodes];

			Resize_Memory(h);
			memset(memory, 0, sizeof(float) * h * layer[0]->time_step * layer[0]->number_nodes);

			// copy x_test to neuron
			for (int i = h; --i >= 0;) {
				memcpy(&memory[i * layer[0]->time_step * layer[0]->number_nodes], x_batch[i], sizeof(float) * ((sequence_length_batch) ? (sequence_length_batch[i]) : (layer[0]->time_step)) * layer[0]->number_nodes);
			}
			cudaMemcpy(layer[0]->neuron, memory, sizeof(float) * this->batch_size * layer[0]->time_step * layer[0]->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;

			// forward propagation
			for (int i = 1; i < layer.size(); i++) {
				for (int t = 0; t < layer[i]->time_step; t++) {
					layer[i]->Forward(t);
					layer[i]->Activate(t);
				}
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch, &hypothesis[g - h + 1], sequence_length_batch);
			h = 0;
		}
	}
	if (x_batch) {
		delete[] x_batch;
	}
	if (y_batch) {
		delete[] y_batch;
	}
	if (sequence_length_batch) {
		delete[] sequence_length_batch;
	}

	return loss / test_size;
}
double Neural_Networks::Fit(float **x_train, float **y_train, vector<string> reference[], int sequence_length[], int train_size, int batch_size) {
	int *sequence_length_batch = (sequence_length) ? (new int[batch_size]) : (nullptr);

	float **x_batch = new float*[batch_size];
	float **y_batch = (y_train) ? (new float*[batch_size]) : (nullptr);

	double loss = 0;

	vector<string> *reference_batch = (reference) ? (new vector<string>[batch_size]) : (nullptr);

	for (int g = 0, h = 0; g < train_size; g++) {
		if (x_batch) {
			x_batch[h] = x_train[g];
		}
		if (y_batch) {
			y_batch[h] = y_train[g];
		}
		if (reference_batch) {
			reference_batch[h] = reference[g];
		}
		if (sequence_length_batch) {
			sequence_length_batch[h] = sequence_length[g];
		}

		if (++h == batch_size || g == train_size - 1) {
			// printf("%d\n", g);

			float *memory = new float[h * layer[0]->time_step * layer[0]->number_nodes];

			Resize_Memory(h);
			memset(memory, 0, sizeof(float) * h * layer[0]->time_step * layer[0]->number_nodes);

			// copy x_train to neuron
			while (--h >= 0) {
				memcpy(&memory[h * layer[0]->time_step * layer[0]->number_nodes], x_batch[h], sizeof(float) * ((sequence_length_batch) ? (sequence_length_batch[h]) : (layer[0]->time_step)) * layer[0]->number_nodes);
			}
			cudaMemcpy(layer[0]->neuron, memory, sizeof(float) * this->batch_size * layer[0]->time_step * layer[0]->number_nodes, cudaMemcpyHostToDevice);
			delete[] memory;
			h = 0;

			// add gaussian noise to the neuron if noise specified
			for (int i = 0; i < layer.size(); i++) {
				Layer *layer = this->layer[i];

				if (strstr(layer->properties.c_str(), "noise")) {
					int memory_size = layer->batch_size * layer->time_step * layer->number_nodes;
					int number_blocks = (memory_size / NUMBER_THREADS + 1 > 65536) ? (65536) : (memory_size / NUMBER_THREADS + 1);

					double scale = atof(strstr(layer->properties.c_str(), "noise") + 5);

					Random_Normal << <number_blocks, NUMBER_THREADS >> > (layer->neuron, memory_size, scale, rand(), true);
				}
			}

			// forward propagation
			for (int i = 1; i < layer.size(); i++) {
				for (int t = 0; t < layer[i]->time_step; t++) {
					layer[i]->Forward(t);
					layer[i]->Activate(t, true);
				}
			}

			// calculate loss
			loss += Calculate_Loss(layer.back(), y_batch, reference_batch, sequence_length_batch, true);

			// calculate error && backpropagation
			for (int i = layer.size() - 1; i > 0; i--) {
				for (int t = layer[i]->time_step - 1; t >= 0; t--) {
					if (i == layer.size() - 1 && y_batch) {
						layer[i]->Differentiate(t, this->loss, y_batch);
					}
					else {
						layer[i]->Differentiate(t);
					}
					layer[i]->Backward(t);
				}
			}

			// adjust parameter
			for (int i = 0; i < layer.size(); i++) {
				layer[i]->Adjust_Parameter(iterations);
			}
			iterations++;
		}
	}
	if (x_batch) {
		delete[] x_batch;
	}
	if (y_batch) {
		delete[] y_batch;
	}
	if (reference_batch) {
		delete[] reference_batch;
	}
	if (sequence_length_batch) {
		delete[] sequence_length_batch;
	}
	return loss / train_size;
}

Neural_Networks::Neural_Networks() {
	batch_size = 1;
	loss = nullptr;
	optimizer = nullptr;
}
Neural_Networks::~Neural_Networks() {
	if (loss) {
		delete loss;
	}
	if (optimizer) {
		optimizer->Destruct();
		delete optimizer;
	}
	for (int i = 0; i < layer.size(); i++) {
		delete layer[i];
	}
}

void Neural_Networks::Compile(Loss loss, Optimizer optimizer) {
	if (this->optimizer) {
		this->optimizer->Destruct();
		delete this->optimizer;
	}
	this->iterations = 0;
	this->loss = loss.Copy();
	this->optimizer = optimizer.Copy();

	for (int i = 0; i < layer.size(); i++) {
		layer[i]->Compile(optimizer.Copy());
	}
}
void Neural_Networks::Load_Parameters(string path) {
	ifstream file(path);

	if (file.is_open()) {
		for (int i = 0; i < layer.size(); i++) {
			Layer *layer = this->layer[i];

			if (layer->batch_normalization) {
				layer->batch_normalization->Load(file);
			}
			if (layer->bias) {
				float *memory = new float[layer->number_maps];

				for (int j = 0; j < layer->number_maps; j++) file >> memory[j];
				cudaMemcpy(layer->bias, memory, sizeof(float) * layer->number_maps, cudaMemcpyHostToDevice);

				delete[] memory;
			}
		}
		for (int i = 0; i < layer.size(); i++) {
			for (int j = 0; j < layer[i]->connection.size(); j++) {
				Connection *connection = layer[i]->connection[j];

				if (connection->properties[0] == 'W') {
					float *memory = new float[connection->number_weights];

					for (int j = 0; j < connection->number_weights; j++) file >> memory[j];
					cudaMemcpy(connection->weight, memory, sizeof(float) * connection->number_weights, cudaMemcpyHostToDevice);

					delete[] memory;
				}
			}
		}
		file.close();
	}
	else {
		cerr << "[Load_Weights], " + path + " not found" << endl;
	}
}
void Neural_Networks::Save_Parameters(string path) {
	ofstream file(path);

	for (int i = 0; i < layer.size(); i++) {
		Layer *layer = this->layer[i];

		if (layer->batch_normalization) {
			layer->batch_normalization->Save(file);
		}
		if (layer->bias) {
			float *memory = new float[layer->number_maps];

			cudaMemcpy(memory, layer->bias, sizeof(float) * layer->number_maps, cudaMemcpyDeviceToHost);
			for (int j = 0; j < layer->number_maps; j++) file << memory[j] << endl;

			delete[] memory;
		}
	}
	for (int i = 0; i < layer.size(); i++) {
		for (int j = 0; j < layer[i]->connection.size(); j++) {
			Connection *connection = layer[i]->connection[j];

			if (connection->properties[0] == 'W') {
				float *memory = new float[connection->number_weights];

				cudaMemcpy(memory, connection->weight, sizeof(float) * connection->number_weights, cudaMemcpyDeviceToHost);
				for (int j = 0; j < connection->number_weights; j++) file << memory[j] << endl;

				delete[] memory;
			}
		}
	}
	file.close();
}
void Neural_Networks::Predict(float input[], float output[]) {
	Predict(&input, &output);
}
void Neural_Networks::Predict(float **input, float **output, int batch_size) {
	float *memory = new float[batch_size * layer.front()->time_step * ((layer.front()->number_nodes > layer.back()->number_nodes) ? (layer.front()->number_nodes) : (layer.back()->number_nodes))];

	Resize_Memory(batch_size);

	for (int h = 0, i = 0; h < batch_size; h++) {
		memcpy(&memory[h * layer[i]->time_step * layer[i]->number_nodes], input[h], sizeof(float) * layer[i]->time_step * layer[i]->number_nodes);
	}
	cudaMemcpy(layer.front()->neuron, memory, sizeof(float) * batch_size * layer.front()->time_step * layer.front()->number_nodes, cudaMemcpyHostToDevice);

	for (int i = 1; i < layer.size(); i++) {
		for (int t = 0; t < layer[i]->time_step; t++) {
			layer[i]->Forward(t);
			layer[i]->Activate(t);
		}
	}
	cudaMemcpy(memory, layer.back()->neuron, sizeof(float) * batch_size * layer.back()->time_step * layer.back()->number_nodes, cudaMemcpyDeviceToHost);

	for (int h = 0, i = layer.size() - 1; h < batch_size; h++) {
		memcpy(output[h], &memory[h * layer[i]->time_step * layer[i]->number_nodes], sizeof(float) * layer[i]->time_step * layer[i]->number_nodes);
	}
	delete[] memory;
}

int* Neural_Networks::Shuffle(int *data, int data_size, int seed) {
	srand(seed);

	for (int i = 0, index; i < data_size; i++) {
		int d = data[index = rand() % data_size];

		data[index] = data[i];
		data[i] = d;
	}
	return data;
}

float** Neural_Networks::Shuffle(float **data, int data_size, int seed) {
	srand(seed);

	for (int i = 0, index; i < data_size; i++) {
		float *d = data[index = rand() % data_size];

		data[index] = data[i];
		data[i] = d;
	}
	return data;
}

double Neural_Networks::Evaluate(float **x_test, float **y_test, int test_size, int batch_size) {
	return Evaluate(x_test, y_test, nullptr, nullptr, test_size, batch_size);
}
double Neural_Networks::Evaluate(float **x_test, vector<string> hypothesis[], int sequence_length[], int test_size, int batch_size) {
	return Evaluate(x_test, nullptr, hypothesis, sequence_length, test_size, batch_size);
}
double Neural_Networks::Fit(float **x_train, float **y_train, int train_size, int batch_size) {
	return Fit(x_train, y_train, nullptr, nullptr, train_size, batch_size);
}
double Neural_Networks::Fit(float **x_train, vector<string> reference[], int sequence_length[], int train_size, int batch_size) {
	return Fit(x_train, nullptr, reference, sequence_length, train_size, batch_size);
}

vector<string>* Neural_Networks::Shuffle(vector<string> *data, int data_size, int seed) {
	srand(seed);

	for (int i = 0, index; i < data_size; i++) {
		vector<string> d = data[index = rand() % data_size];

		data[index] = data[i];
		data[i] = d;
	}
	return data;
}

Connection* Neural_Networks::Connect(int from, int to, string properties, unordered_multimap<int, int> *channel_connection, unordered_multimap<int, int> *time_connection) {
	Connection *connection = new Connection(layer[from], layer[to], properties, channel_connection, time_connection);

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

	if (layer[from]->lstm) {
		for (int i = 1; i < LSTM::number_weight_types; i++) {
			Connection *c = connection->Copy(i);

			layer[from]->connection.push_back(c);
			layer[to]->child_connection.push_back(c);
		}
	}
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

LSTM* Neural_Networks::Add(LSTM LSTM) {
	Layer *layer = new Layer(LSTM.time_step, LSTM.number_maps, LSTM.map_width, LSTM.map_height, LSTM.map_depth, LSTM.properties);

	layer->lstm = LSTM.Copy(layer);
	this->layer.push_back(layer);
	return layer->lstm;
}

RNN* Neural_Networks::Add(RNN RNN) {
	Layer *layer = new Layer(RNN.time_step, RNN.number_maps, RNN.map_width, RNN.map_height, RNN.map_depth, RNN.properties);

	layer->rnn = RNN.Copy(layer);
	this->layer.push_back(layer);
	return layer->rnn;
}