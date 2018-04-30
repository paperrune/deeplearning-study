#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <unordered_map>

#include "Neural_Networks.h"

#define NUMBER_THREADS 64

class LSTM_Optimizer {
public:
	Optimizer optimizer[4];
};

__device__ float Calculate_Gradient(int parameter_index, double _gradient, double learning_rate, Optimizer &optimizer, bool update = false) {
	int type = optimizer.type;

	double epsilon = optimizer.epsilon;
	double factor_1 = optimizer.factor_1;
	double factor_2 = optimizer.factor_2;

	float &gradient = optimizer.gradient[parameter_index];

	if (type == 0) {
		return (gradient = -learning_rate * _gradient);
	}

	float &velocity = optimizer.velocity[parameter_index];

	if (type == 1) { // momentum
		double v = factor_1 * velocity - learning_rate * _gradient;

		if (update) {
			velocity = v;
		}
		return (gradient = v);
	}
	if (type == 2) { // nesterov
		double v = factor_1 * velocity - learning_rate * _gradient;

		if (update) {
			velocity = factor_1 * v - learning_rate * _gradient;
		}
		return (gradient = factor_1 * v - learning_rate * _gradient);
	}
	if (type == 3) { // adagrad
		double c = velocity + _gradient * _gradient;

		if (update) {
			velocity = c;
		}
		return (gradient = -learning_rate * _gradient / sqrt(c + epsilon));
	}
	if (type == 4) { // rmsprop
		double c = factor_1 * velocity + (1 - factor_1) * _gradient * _gradient;

		if (update) {
			velocity = c;
		}
		return (gradient = -learning_rate * _gradient / sqrt(c + epsilon));
	}

	float &momentum = optimizer.momentum[parameter_index];

	if (type == 5) { // adadelta
		double c = factor_1 * velocity + (1 - factor_1) * _gradient * _gradient;

		gradient = -sqrt(momentum + epsilon) / sqrt(c + epsilon) * _gradient;

		if (update) {
			momentum = factor_1 * momentum + (1 - factor_1) * gradient * gradient;
			velocity = c;
		}
		return gradient;

	}
	if (type == 6) { // adam
		double m = factor_1 * momentum + (1 - factor_1) * _gradient;
		double v = factor_2 * velocity + (1 - factor_2) * _gradient * _gradient;

		if (momentum == 0 && velocity == 0) {
			m = m / (1 - factor_1);
			v = v / (1 - factor_2);
		}
		if (update) {
			momentum = m;
			velocity = v;
		}
		return (gradient = -learning_rate * m / sqrt(v + epsilon));
	}
	return 0;
}

__global__ void Activate(bool training, int time_index, float _neuron[], Batch_Normalization batch_normalization) {
	int j = blockIdx.x;
	int t = time_index;

	int batch_size = batch_normalization.batch_size;
	int map_size = batch_normalization.map_size;
	int number_maps = batch_normalization.number_maps;
	int number_nodes = batch_normalization.number_nodes;
	int time_step = batch_normalization.time_step;

	float *gamma = batch_normalization.gamma;
	float *beta = batch_normalization.beta;
	float *mean = &batch_normalization.mean[t * number_maps];
	float *variance = &batch_normalization.variance[t * number_maps];

	double epsilon = batch_normalization.epsilon;

	if (training) {
		float *sum_mean = &batch_normalization.sum_mean[t * number_maps];
		float *sum_variance = &batch_normalization.sum_variance[t * number_maps];

		float *neuron = &_neuron[t * number_nodes + j * map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[t * number_nodes + j * map_size];
		float *neuron_normalized = &batch_normalization.neuron_normalized[t * number_nodes + j * map_size];

		__shared__ double sum[NUMBER_THREADS];
		__shared__ double standard_deviation;

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
			int index = h / map_size * time_step * number_nodes + (h % map_size);

			sum[threadIdx.x] += neuron[index];
		}
		for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
			__syncthreads();

			if (threadIdx.x < h) {
				sum[threadIdx.x] += sum[threadIdx.x + h];
			}
		}
		if (threadIdx.x == 0) {
			sum_mean[j] += (mean[j] = sum[0] / (batch_size * map_size));
		}
		__syncthreads();

		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
			int index = h / map_size * time_step * number_nodes + (h % map_size);

			sum[threadIdx.x] += (neuron[index] - mean[j]) * (neuron[index] - mean[j]);
		}
		for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
			__syncthreads();

			if (threadIdx.x < h) {
				sum[threadIdx.x] += sum[threadIdx.x + h];
			}
		}
		if (threadIdx.x == 0) {
			sum_variance[j] += (variance[j] = sum[0] / (batch_size * map_size));
			standard_deviation = sqrt(variance[j] + epsilon);
		}
		__syncthreads();

		for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
			int index = h / map_size * time_step * number_nodes + (h % map_size);

			neuron_backup[index] = neuron[index];
			neuron_normalized[index] = (neuron[index] - mean[j]) / standard_deviation;
			neuron[index] = gamma[j] * neuron_normalized[index] + beta[j];
		}
	}
	else {
		float *neuron = &_neuron[t * number_nodes + j * map_size];
		float *neuron_backup = &batch_normalization.neuron_backup[t * number_nodes + j * map_size];

		double standard_deviation = sqrt(variance[j] + epsilon);

		for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
			int index = h / map_size * time_step * number_nodes + (h % map_size);

			neuron_backup[index] = neuron[index];
			neuron[index] = gamma[j] / standard_deviation * neuron[index] + (beta[j] - gamma[j] * mean[j] / standard_deviation);
		}
	}
}
__global__ void Activate(int option, int time_index, Layer layer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int k = j / layer.map_size;
		int h = i / layer.number_nodes;

		int index = (h * layer.time_step + t) * layer.number_nodes + j;

		if (option == 0) {
			if (layer.neuron[1]) {
				layer.neuron[0][index] += layer.neuron[1][index];
			}
			if (layer.bias) {
				layer.neuron[0][index] += layer.bias[k];
			}
		}
		else if (option == 1) { // ELU
			layer.neuron[0][index] = (layer.neuron[0][index] > 0) ? (layer.neuron[0][index]) : (*layer.slope * (exp(layer.neuron[0][index]) - 1));
		}
		else if (option == 2) { // PReLU
			layer.neuron[2][index] = layer.neuron[0][index];
			layer.neuron[0][index] *= (layer.neuron[2][index] > 0) ? (1) : (layer.slope[j]);
		}
		else if (option == 3) { // ReLU
			layer.neuron[0][index] *= (layer.neuron[0][index] > 0) ? (1) : (*layer.slope);
		}
		else if (option == 4) { // sigmoid
			layer.neuron[0][index] = 1 / (1 + exp(-layer.neuron[0][index]));
		}
		else if (option == 5 && j == 0) { // softmax
			float *neuron = &layer.neuron[0][index];

			double max;
			double sum = 0;

			for (int j = 0; j < layer.number_nodes; j++) {
				if (j == 0 || max < neuron[j]) {
					max = neuron[j];
				}
			}
			for (int j = 0; j < layer.number_nodes; j++) {
				neuron[j] = exp(neuron[j] - max);
				sum += neuron[j];
			}
			for (int j = 0; j < layer.number_nodes; j++) {
				neuron[j] /= sum;
			}
		}
		else if (option == 6) { // tangent
			layer.neuron[0][index] = 2 / (1 + exp(-2 * layer.neuron[0][index])) - 1;
		}
	}
}
__global__ void Activate(int option, int time_index, float neuron_backup[], Layer layer, LSTM_Node LSTM_node, bool backward = false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int k = j / layer.map_size;
		int h = i / layer.number_nodes;

		int index = (h * layer.time_step + t) * layer.number_nodes + j;

		if (option == 0) {
			float *input_bias = LSTM_node.bias[LSTM_node.input];
			float *forget_bias = LSTM_node.bias[LSTM_node.forget];
			float *output_bias = LSTM_node.bias[LSTM_node.output];
			float *cell_bias = LSTM_node.bias[LSTM_node.cell];

			float *input_neuron[] = { LSTM_node.neuron[LSTM_node.input][0], LSTM_node.neuron[LSTM_node.input][1] };
			float *forget_neuron[] = { LSTM_node.neuron[LSTM_node.forget][0], LSTM_node.neuron[LSTM_node.forget][1] };
			float *output_neuron[] = { LSTM_node.neuron[LSTM_node.output][0], LSTM_node.neuron[LSTM_node.output][1] };
			float *cell_neuron[] = { LSTM_node.neuron[LSTM_node.cell][0], LSTM_node.neuron[LSTM_node.cell][1] };
			float *cell_output_neuron = LSTM_node.neuron[LSTM_node.cell_output][0];

			float *previous_cell_output_neuron = (neuron_backup) ? (neuron_backup) : (LSTM_node.neuron[LSTM_node.cell_output][0]);

			input_neuron[0][index] += input_neuron[1][index] + input_bias[k];
			forget_neuron[0][index] += forget_neuron[1][index] + forget_bias[k];
			output_neuron[0][index] += output_neuron[1][index] + output_bias[k];
			cell_neuron[0][index] += cell_neuron[1][index] + cell_bias[k];

			if (backward == false && t) {
				input_neuron[0][index] += LSTM_node.peephole[LSTM_node.input][k] * previous_cell_output_neuron[index - layer.number_nodes];
				forget_neuron[0][index] += LSTM_node.peephole[LSTM_node.forget][k] * previous_cell_output_neuron[index - layer.number_nodes];
				output_neuron[0][index] += LSTM_node.peephole[LSTM_node.output][k] * previous_cell_output_neuron[index - layer.number_nodes];
			}
			else if (backward == true && t + 1 < layer.time_step) {
				input_neuron[0][index] += LSTM_node.peephole[LSTM_node.input][k] * previous_cell_output_neuron[index + layer.number_nodes];
				forget_neuron[0][index] += LSTM_node.peephole[LSTM_node.forget][k] * previous_cell_output_neuron[index + layer.number_nodes];
				output_neuron[0][index] += LSTM_node.peephole[LSTM_node.output][k] * previous_cell_output_neuron[index + layer.number_nodes];
			}
			input_neuron[0][index] = 1 / (1 + exp(-input_neuron[0][index]));
			forget_neuron[0][index] = 1 / (1 + exp(-forget_neuron[0][index]));
			output_neuron[0][index] = 1 / (1 + exp(-output_neuron[0][index]));
			cell_neuron[0][index] = 2 / (1 + exp(-2 * cell_neuron[0][index])) - 1;

			if (backward == false && t) {
				cell_output_neuron[index] = forget_neuron[0][index] * previous_cell_output_neuron[index - layer.number_nodes];
			}
			else if (backward == true && t + 1 < layer.time_step) {
				cell_output_neuron[index] = forget_neuron[0][index] * previous_cell_output_neuron[index + layer.number_nodes];
			}
			else {
				cell_output_neuron[index] = 0;
			}
			cell_output_neuron[index] += input_neuron[0][index] * cell_neuron[0][index];
		}
		else if (option == 1) {
			layer.neuron[0][index] = LSTM_node.neuron[LSTM_node.output][0][index] * (2 / (1 + exp(-2 * LSTM_node.neuron[LSTM_node.cell_output][0][index])) - 1);
		}
	}
}
__global__ void Adjust_Parameter(int number_parameters, float parameter[], double gradient_clip, double learning_rate, Optimizer optimizer) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < number_parameters; j += gridDim.x * blockDim.x) {
		parameter[j] += Calculate_Gradient(j, optimizer.gradient[j], gradient_clip * learning_rate, optimizer, true);
	}
}
__global__ void Backpropagate(int option, int time_index, Connection connection, Layer layer, Layer parent_layer, bool backward = false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * parent_layer.number_nodes) {
		int j = i % parent_layer.number_nodes;
		int h = i / parent_layer.number_nodes;

		if (option == 0) {
			int m = 0;

			float sum = 0, *error = &layer.error[0][(h * layer.time_step + t) * layer.number_nodes];

			for (int l = j; (&connection.from_errors[l])->weight != -1; l += parent_layer.number_nodes, m++) {
				sum += error[(&connection.from_errors[l])->next_node];
			}
			parent_layer.error[0][(h * layer.time_step + t) * parent_layer.number_nodes + j] += sum;
		}
		else if (option == 1) {
			float sum = 0, *error = &layer.error[0][(h * layer.time_step + t) * layer.number_nodes];

			for (int l = j; (&connection.from_errors[l])->weight != -1; l += parent_layer.number_nodes) {
				sum += error[(&connection.from_errors[l])->next_node] * connection.weight[(&connection.from_errors[l])->weight];
			}
			parent_layer.error[0][(h * layer.time_step + t) * parent_layer.number_nodes + j] += sum;
		}
		else if (option == 2) {
			float sum = 0, *error = &layer.error[1][(h * layer.time_step + t) * layer.number_nodes];

			for (int l = j; (&connection.from_errors[l])->weight != -1; l += parent_layer.number_nodes) {
				sum += error[(&connection.from_errors[l])->next_node] * connection.weight[(&connection.from_errors[l])->weight];
			}
			parent_layer.error[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes + j] += sum;
		}
		else if (option == 3) {
			parent_layer.error[0][(h * layer.time_step + t) * parent_layer.number_nodes + j] += layer.error[0][(h * layer.time_step + t) * layer.number_nodes + j];
		}
		else if (option == 4) {
			parent_layer.error[0][(h * layer.time_step + t) * parent_layer.number_nodes + j] += layer.error[0][(h * layer.time_step + t) * layer.number_nodes + (&connection.from_errors[j])->next_node];
		}
	}
}
__global__ void Backpropagate(int option, int time_index, Connection connection, Layer layer, Layer parent_layer, LSTM_Node LSTM_node, LSTM_Weight LSTM_weight, bool backward = false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * parent_layer.number_nodes) {
		int j = i % parent_layer.number_nodes;
		int h = i / parent_layer.number_nodes;

		if (option == 0) {
			for (int i = 0; i < LSTM_weight.number_weight_types; i++) {
				float sum = 0, *error = &LSTM_node.error[i][0][(h * layer.time_step + t) * layer.number_nodes];

				for (int l = j; (&connection.from_errors[l])->weight != -1; l += parent_layer.number_nodes) {
					sum += error[(&connection.from_errors[l])->next_node] * LSTM_weight.weight[i][(&connection.from_errors[l])->weight];
				}
				parent_layer.error[0][(h * layer.time_step + t) * parent_layer.number_nodes + j] += sum;
			}
		}
		else if (option == 1) {
			for (int i = 0; i < LSTM_weight.number_weight_types; i++) {
				float sum = 0, *error = &LSTM_node.error[i][1][(h * layer.time_step + t) * layer.number_nodes];

				for (int l = j; (&connection.from_errors[l])->weight != -1; l += parent_layer.number_nodes) {
					sum += error[(&connection.from_errors[l])->next_node] * LSTM_weight.weight[i][(&connection.from_errors[l])->weight];
				}
				parent_layer.error[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes + j] += sum;
			}
		}
	}
}
__global__ void Calculate_Gradient(float gradient[], double learning_rate, Batch_Normalization batch_normalization, Optimizer gamma_optimizer, Optimizer beta_optimizer) {
	int j = blockIdx.x;

	int batch_size = batch_normalization.batch_size;
	int map_size = batch_normalization.map_size;
	int number_nodes = batch_normalization.number_nodes;
	int time_step = batch_normalization.time_step;

	float *error_backup = &batch_normalization.error_backup[j * map_size];
	float *neuron_normalized = &batch_normalization.neuron_normalized[j * map_size];

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_size * time_step; h += blockDim.x) {
		int index = h * number_nodes;

		for (int k = 0; k < map_size; k++) {
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
		float g = Calculate_Gradient(j, sum[0], learning_rate, gamma_optimizer);

		gamma_optimizer.gradient[j] = sum[0];
		gradient[j] = g * g;
	}

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_size * time_step; h += blockDim.x) {
		int index = h * number_nodes;

		for (int k = 0; k < map_size; k++) {
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
		float g = Calculate_Gradient(j, sum[0], learning_rate, beta_optimizer);

		beta_optimizer.gradient[j] = sum[0];
		gradient[j] += g * g;
	}
}
__global__ void Calculate_Gradient(float gradient[], double learning_rate, Connection connection, Layer layer, Layer parent_layer, Optimizer optimizer, bool recurrent, bool backward) {
	for (int k = blockIdx.x; k < connection.number_weights; k += gridDim.x) {
		__shared__ double sum[NUMBER_THREADS];

		sum[threadIdx.x] = 0;
		for (int s = threadIdx.x; s < layer.batch_size * layer.time_step; s += blockDim.x) {
			int t = s % layer.time_step;
			int h = s / layer.time_step;

			if (recurrent == false) {
				if (parent_layer.time_mask_device == nullptr || parent_layer.time_mask_device[t]) {
					float *error = &layer.error[0][(h * layer.time_step + t) * layer.number_nodes];
					float *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

					for (int l = k; (&connection.from_weights[l])->weight != -1; l += connection.number_weights) {
						sum[threadIdx.x] += error[(&connection.from_weights[l])->next_node] * neuron[(&connection.from_weights[l])->prev_node];
					}
				}
			}
			else if ((backward == false && t) || (backward == true && t + 1 < layer.time_step)) {
				float *error = &layer.error[1][(h * layer.time_step + t) * layer.number_nodes];
				float *neuron = &parent_layer.neuron[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes];

				for (int l = k; (&connection.from_weights[l])->weight != -1; l += connection.number_weights) {
					sum[threadIdx.x] += error[(&connection.from_weights[l])->next_node] * neuron[(&connection.from_weights[l])->prev_node];
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
			float g = Calculate_Gradient(k, sum[0], learning_rate, optimizer);

			optimizer.gradient[k] = sum[0];
			gradient[k] = g * g;
		}
	}
}
__global__ void Calculate_Gradient(float gradient[], double learning_rate, Connection connection, Layer layer, Layer parent_layer, LSTM_Node LSTM_node, LSTM_Weight LSTM_weight, LSTM_Optimizer optimizer, bool recurrent, bool backward) {
	for (int k = blockIdx.x; k < LSTM_weight.number_weight_types * connection.number_weights; k += gridDim.x) {
		int i = k / connection.number_weights;
		int j = k % connection.number_weights;

		__shared__ double sum[NUMBER_THREADS];

		sum[threadIdx.x] = 0;
		for (int s = threadIdx.x; s < layer.batch_size * layer.time_step; s += blockDim.x) {
			int t = s % layer.time_step;
			int h = s / layer.time_step;

			if (recurrent == false) {
				if (parent_layer.time_mask_device == nullptr || parent_layer.time_mask_device[t]) {
					float *error = &LSTM_node.error[i][0][(h * layer.time_step + t) * layer.number_nodes];
					float *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

					for (int l = j; (&connection.from_weights[l])->weight != -1; l += connection.number_weights) {
						sum[threadIdx.x] += error[(&connection.from_weights[l])->next_node] * neuron[(&connection.from_weights[l])->prev_node];
					}
				}
			}
			else if ((backward == false && t) || (backward == true && t + 1 < layer.time_step)) {
				float *error = &LSTM_node.error[i][1][(h * layer.time_step + t) * layer.number_nodes];
				float *neuron = &parent_layer.neuron[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes];

				for (int l = j; (&connection.from_weights[l])->weight != -1; l += connection.number_weights) {
					sum[threadIdx.x] += error[(&connection.from_weights[l])->next_node] * neuron[(&connection.from_weights[l])->prev_node];
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
			float g = Calculate_Gradient(j, sum[0], learning_rate, optimizer.optimizer[i]);

			optimizer.optimizer[i].gradient[j] = sum[0];
			gradient[k] = g * g;
		}
	}
}
__global__ void Calculate_Gradient(float gradient[], double learning_rate, Layer layer, Optimizer optimizer) {
	int j = blockIdx.x;

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < layer.batch_size * layer.time_step; h += blockDim.x) {
		int index = h * layer.number_nodes + j;

		sum[threadIdx.x] += (layer.neuron[2][index] > 0) ? (0) : (layer.error[0][index] * layer.neuron[2][index]);
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		float g = Calculate_Gradient(j, sum[0], learning_rate, optimizer);

		optimizer.gradient[j] = sum[0];
		gradient[j] = g * g;
	}
}
__global__ void Calculate_Gradient(float gradient[], float error_backup[], double learning_rate, Layer layer, Optimizer optimizer) {
	int j = blockIdx.x;

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < layer.batch_size * layer.time_step; h += blockDim.x) {
		int index = h * layer.number_nodes + j * layer.map_size;

		for (int k = 0; k < layer.map_size; k++) {
			sum[threadIdx.x] += (error_backup) ? (error_backup[index + k]) : (layer.error[0][index + k]);
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		float g = Calculate_Gradient(j, sum[0], learning_rate, optimizer);

		optimizer.gradient[j] = sum[0];
		gradient[j] = g * g;
	}
}
__global__ void Calculate_Gradient(float gradient[], float error_backup[], float neuron_backup[], double learning_rate, Layer layer, LSTM_Node LSTM_node, LSTM_Optimizer bias_optimizer, LSTM_Optimizer peephole_optimizer, bool backward, bool batch_normalization = false) {
	int i = blockIdx.x / layer.number_maps;
	int j = blockIdx.x % layer.number_maps;

	float *previous_cell_output_neuron = (neuron_backup) ? (neuron_backup) : (LSTM_node.neuron[LSTM_node.cell_output][0]);

	__shared__ double sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < layer.batch_size * layer.time_step; h += blockDim.x) {
		int index = h * layer.number_nodes + j * layer.map_size;

		for (int k = 0; k < layer.map_size; k++) {
			sum[threadIdx.x] += (error_backup) ? (error_backup[index + k]) : (LSTM_node.error[i][0][index + k]);
		}
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		float g = Calculate_Gradient(j, sum[0], learning_rate, bias_optimizer.optimizer[i]);

		bias_optimizer.optimizer[i].gradient[j] = sum[0];
		gradient[blockIdx.x] = g * g;
	}

	if (i < LSTM_node.number_weight_types - 1) {
		sum[threadIdx.x] = 0;
		for (int h = threadIdx.x; h < layer.batch_size * layer.time_step; h += blockDim.x) {
			int index = h * layer.number_nodes + j * layer.map_size;

			if (backward == false && h % layer.time_step) {
				for (int k = 0; k < layer.map_size; k++) {
					sum[threadIdx.x] += ((error_backup) ? (error_backup[index + k]) : (LSTM_node.error[i][0][index + k])) * previous_cell_output_neuron[index + k - layer.number_nodes];
				}
			}
			else if (backward == true && (h % layer.time_step) + 1 < layer.time_step) {
				for (int k = 0; k < layer.map_size; k++) {
					sum[threadIdx.x] += ((error_backup) ? (error_backup[index + k]) : (LSTM_node.error[i][0][index + k])) * previous_cell_output_neuron[index + k + layer.number_nodes];
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
			float g = Calculate_Gradient(j, sum[0], learning_rate, peephole_optimizer.optimizer[i]);

			peephole_optimizer.optimizer[i].gradient[j] = sum[0];
			gradient[blockIdx.x] += g * g;
		}
	}
}
__global__ void Differentiate(int time_index, float _error[], Batch_Normalization batch_normalization) {
	int j = blockIdx.x;
	int t = time_index;

	int batch_size = batch_normalization.batch_size;
	int map_size = batch_normalization.map_size;
	int number_maps = batch_normalization.number_maps;
	int number_nodes = batch_normalization.number_nodes;
	int time_step = batch_normalization.time_step;

	float *gamma = batch_normalization.gamma;
	float *mean = &batch_normalization.mean[t * number_maps];
	float *variance = &batch_normalization.variance[t * number_maps];

	float *error = &_error[t * number_nodes + j * map_size];
	float *error_backup = &batch_normalization.error_backup[t * number_nodes + j * map_size];
	float *error_normalized = &batch_normalization.error_normalized[t * number_nodes + j * map_size];
	float *neuron_backup = &batch_normalization.neuron_backup[t * number_nodes + j * map_size];

	__shared__ double sum[NUMBER_THREADS];
	__shared__ double error_mean;
	__shared__ double error_variance;

	double epsilon = batch_normalization.epsilon;
	double standard_deviation = sqrt(variance[j] + epsilon);

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
		int index = h / map_size * time_step * number_nodes + (h % map_size);

		error_normalized[index] = error[index] * gamma[j];
		sum[threadIdx.x] += error_normalized[index] * (neuron_backup[index] - mean[j]);
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		error_variance = sum[0] * (-0.5) * pow(variance[j] + epsilon, (double)-1.5);
	}
	__syncthreads();

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
		int index = h / map_size * time_step * number_nodes + (h % map_size);

		sum[threadIdx.x] += error_normalized[index];
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		error_mean = -sum[0] / standard_deviation;
	}
	__syncthreads();

	for (int h = threadIdx.x; h < batch_size * map_size; h += blockDim.x) {
		int index = h / map_size * time_step * number_nodes + (h % map_size);

		error_backup[index] = error[index];
		error[index] = error_normalized[index] / standard_deviation + error_variance * 2 * (neuron_backup[index] - mean[j]) / (batch_size * map_size) + error_mean / (batch_size * map_size);
	}
}
__global__ void Differentiate(int loss_type, int time_index, float loss[], float target_output[], Layer layer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int h = i / layer.number_nodes;

		int index = (h * layer.time_step + t) * layer.number_nodes + j;

		// calculate error
		layer.error[0][index] = layer.neuron[0][index] - target_output[index];

		// calculate loss
		if (loss_type == 0) {
			loss[i] = -(target_output[index] * log(layer.neuron[0][index] + 0.000001) + (1 - target_output[index]) * log(1 - layer.neuron[0][index] + 0.000001));
		}
		else {
			loss[i] = 0.5 * (layer.neuron[0][index] - target_output[index]) * (layer.neuron[0][index] - target_output[index]);
		}
	}
}
__global__ void Differentiate(int option, int time_index, Layer layer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int h = i / layer.number_nodes;

		int index = (h * layer.time_step + t) * layer.number_nodes + j;

		if (option == 0) { // ELU
			layer.error[0][index] *= (layer.neuron[0][index] > 0) ? (1) : (layer.neuron[0][index] + *layer.slope);
		}
		else if (option == 1) { // PReLU
			layer.error[0][index] *= (layer.neuron[2][index] > 0) ? (1) : (layer.slope[j]);
		}
		else if (option == 2) { // ReLU
			layer.error[0][index] *= (layer.neuron[0][index] > 0) ? (1) : (*layer.slope);
		}
		else if (option == 3) { // sigmoid
			layer.error[0][index] *= (1 - layer.neuron[0][index]) * layer.neuron[0][index];
		}
		else if (option == 4) { // tangent
			layer.error[0][index] *= (1 - layer.neuron[0][index]) * (1 + layer.neuron[0][index]);
		}
	}
}
__global__ void Differentiate(int option, int time_index, Layer layer, LSTM_Node LSTM_node, bool backward = false, float input_error_backup[] = nullptr, float forget_error_backup[] = nullptr, float output_error_backup[] = nullptr, float cell_output_neuron_backup[] = nullptr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int k = j / layer.map_size;
		int h = i / layer.number_nodes;

		int index = (h * layer.time_step + t) * layer.number_nodes + j;

		if (option == 0) {
			double active_cell_output_neuron = 2 / (1 + exp(-2 * LSTM_node.neuron[LSTM_node.cell_output][0][index])) - 1;

			LSTM_node.error[LSTM_node.output][0][index] = layer.error[0][index] * active_cell_output_neuron;
			LSTM_node.error[LSTM_node.cell_output][0][index] = layer.error[0][index] * LSTM_node.neuron[LSTM_node.output][0][index] * (1 - active_cell_output_neuron) * (1 + active_cell_output_neuron);
		}
		else if (option == 1) {
			float *input_error[] = { LSTM_node.error[LSTM_node.input][0], LSTM_node.error[LSTM_node.input][1] };
			float *forget_error[] = { LSTM_node.error[LSTM_node.forget][0], LSTM_node.error[LSTM_node.forget][1] };
			float *output_error[] = { LSTM_node.error[LSTM_node.output][0], LSTM_node.error[LSTM_node.output][1] };
			float *cell_error[] = { LSTM_node.error[LSTM_node.cell][0], LSTM_node.error[LSTM_node.cell][1] };
			float *cell_output_error = LSTM_node.error[LSTM_node.cell_output][0];

			float *input_neuron = LSTM_node.neuron[LSTM_node.input][0];
			float *forget_neuron = LSTM_node.neuron[LSTM_node.forget][0];
			float *output_neuron = LSTM_node.neuron[LSTM_node.output][0];
			float *cell_neuron = LSTM_node.neuron[LSTM_node.cell][0];
			float *cell_output_neuron = LSTM_node.neuron[LSTM_node.cell_output][0];

			float *next_input_error = (input_error_backup) ? (input_error_backup) : (input_error[0]);
			float *next_forget_error = (forget_error_backup) ? (forget_error_backup) : (forget_error[0]);
			float *next_output_error = (output_error_backup) ? (output_error_backup) : (output_error[0]);
			float *previous_cell_output_neuron = (cell_output_neuron_backup) ? (cell_output_neuron_backup) : (cell_output_neuron);

			if (backward == false && t + 1 < layer.time_step) {
				cell_output_error[index] += next_input_error[index + layer.number_nodes] * LSTM_node.peephole[LSTM_node.input][k];
				cell_output_error[index] += next_forget_error[index + layer.number_nodes] * LSTM_node.peephole[LSTM_node.forget][k];
				cell_output_error[index] += next_output_error[index + layer.number_nodes] * LSTM_node.peephole[LSTM_node.output][k];
				cell_output_error[index] += cell_output_error[index + layer.number_nodes] * forget_neuron[index + layer.number_nodes];
			}
			else if (backward == true && t) {
				cell_output_error[index] += next_input_error[index - layer.number_nodes] * LSTM_node.peephole[LSTM_node.input][k];
				cell_output_error[index] += next_forget_error[index - layer.number_nodes] * LSTM_node.peephole[LSTM_node.forget][k];
				cell_output_error[index] += next_output_error[index - layer.number_nodes] * LSTM_node.peephole[LSTM_node.output][k];
				cell_output_error[index] += cell_output_error[index - layer.number_nodes] * forget_neuron[index - layer.number_nodes];
			}
			input_error[0][index] = cell_output_error[index] * cell_neuron[index];

			if (backward == false && t) {
				forget_error[0][index] = cell_output_error[index] * previous_cell_output_neuron[index - layer.number_nodes];
			}
			else if (backward == true && t + 1 < layer.time_step) {
				forget_error[0][index] = cell_output_error[index] * previous_cell_output_neuron[index + layer.number_nodes];
			}
			else {
				forget_error[0][index] = 0;
			}
			cell_error[0][index] = cell_output_error[index] * input_neuron[index];

			input_error[0][index] *= (1 - input_neuron[index]) * input_neuron[index];
			input_error[1][index] = input_error[0][index];

			forget_error[0][index] *= (1 - forget_neuron[index]) * forget_neuron[index];
			forget_error[1][index] = forget_error[0][index];

			output_error[0][index] *= (1 - output_neuron[index]) * output_neuron[index];
			output_error[1][index] = output_error[0][index];

			cell_error[0][index] *= (1 - cell_neuron[index]) * (1 + cell_neuron[index]);
			cell_error[1][index] = cell_error[0][index];
		}
	}
}
__global__ void Dropout(bool training, int time_index, double rate, Layer layer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int h = i / layer.number_maps;
	int j = i % layer.number_maps;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int index = (h * layer.time_step + t) * layer.number_nodes + j * layer.map_size;

		if (training) {
			for (int k = 0; k < layer.map_size; k++) {
				if (layer.dropout_mask[h * layer.number_maps + j] == false) {
					layer.neuron[0][index + k] = 0;
				}
			}
		}
		else {
			layer.neuron[0][index] *= rate;
		}
	}
}
__global__ void Feedforward(int option, int time_index, Connection connection, Layer layer, Layer parent_layer, bool backward = false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int h = i / layer.number_nodes;

		if (option == 0) {
			int number_connections = 0;

			float sum = 0, *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

			for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes, number_connections++) {
				sum += neuron[(&connection.from_neurons[l])->prev_node];
			}
			layer.neuron[0][(h * layer.time_step + t) * layer.number_nodes + j] = sum / number_connections;
		}
		else if (option == 1) {
			float max = 0, *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

			for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes) {
				if (max < neuron[(&connection.from_neurons[l])->prev_node]) {
					max = neuron[(&connection.from_neurons[l])->prev_node];
				}
			}
			layer.neuron[0][(h * layer.time_step + t) * layer.number_nodes + j] = max;
		}
		else if (option == 2) {
			float sum = 0, *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

			for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes) {
				sum += neuron[(&connection.from_neurons[l])->prev_node] * connection.weight[(&connection.from_neurons[l])->weight];
			}
			layer.neuron[0][(h * layer.time_step + t) * layer.number_nodes + j] += sum;
		}
		else if (option == 3) {
			float sum = 0, *neuron = &parent_layer.neuron[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes];

			for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes) {
				sum += neuron[(&connection.from_neurons[l])->prev_node] * connection.weight[(&connection.from_neurons[l])->weight];
			}
			layer.neuron[1][(h * layer.time_step + t) * layer.number_nodes + j] += sum;
		}
		else if (option == 4) {
			layer.neuron[0][(h * layer.time_step + t) * layer.number_nodes + j] += parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes + j];
		}
		else if (option == 5) {
			layer.neuron[0][(h * layer.time_step + t) * layer.number_nodes + j] = parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes + (&connection.from_neurons[j])->prev_node];
		}
	}
}
__global__ void Feedforward(int option, int time_index, Connection connection, Layer layer, Layer parent_layer, LSTM_Node LSTM_node, LSTM_Weight LSTM_weight, bool backward = false) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = time_index;

	if (i < layer.batch_size * layer.number_nodes) {
		int j = i % layer.number_nodes;
		int h = i / layer.number_nodes;

		if (option == 0) {
			for (int i = 0; i < LSTM_weight.number_weight_types; i++) {
				float sum = 0, *neuron = &parent_layer.neuron[0][(h * layer.time_step + t) * parent_layer.number_nodes];

				for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes) {
					sum += neuron[(&connection.from_neurons[l])->prev_node] * LSTM_weight.weight[i][(&connection.from_neurons[l])->weight];
				}
				LSTM_node.neuron[i][0][(h * layer.time_step + t) * layer.number_nodes + j] += sum;
			}
		}
		else if (option == 1) {
			for (int i = 0; i < LSTM_weight.number_weight_types; i++) {
				float sum = 0, *neuron = &parent_layer.neuron[0][((backward == false) ? (h * layer.time_step + t - 1) : (h * layer.time_step + t + 1)) * parent_layer.number_nodes];

				for (int l = j; (&connection.from_neurons[l])->weight != -1; l += layer.number_nodes) {
					sum += neuron[(&connection.from_neurons[l])->prev_node] * LSTM_weight.weight[i][(&connection.from_neurons[l])->weight];
				}
				LSTM_node.neuron[i][1][(h * layer.time_step + t) * layer.number_nodes + j] += sum;
			}
		}
	}
}

__global__ void Add(int memory_size, float input[], float operand[], float output[]) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		output[j] = input[j] + operand[j];
	}
}
__global__ void Random_Normal(int memory_size, float memory[], double scale, int seed = 0) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		curandState s[NUMBER_THREADS];

		curand_init(seed + j, 0, 0, &s[threadIdx.x]);
		memory[j] = scale * curand_normal(&s[threadIdx.x]);
	}
}
__global__ void Random_Uniform(int memory_size, float memory[], double scale, int seed = 0) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		curandState s[NUMBER_THREADS];

		curand_init(seed + j, 0, 0, &s[threadIdx.x]);
		memory[j] = scale * (2 * curand_uniform(&s[threadIdx.x]) - 1);
	}
}
__global__ void Set(int memory_size, float memory[], float value) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		memory[j] = value;
	}
}
__global__ void Merge(int memory_size, float memory[]) {
	__shared__ float sum[NUMBER_THREADS];

	sum[threadIdx.x] = 0;
	for (int h = threadIdx.x; h < memory_size; h += blockDim.x) {
		sum[threadIdx.x] += memory[h];
	}
	for (int h = (blockDim.x >> 1); h; h = (h >> 1)) {
		__syncthreads();

		if (threadIdx.x < h) {
			sum[threadIdx.x] += sum[threadIdx.x + h];
		}
	}
	if (threadIdx.x == 0) {
		memory[0] = sum[0];
	}
}
__global__ void Multiply(int memory_size, float input[], double operand, float output[]) {
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < memory_size; j += gridDim.x * blockDim.x) {
		output[j] = input[j] * operand;
	}
}

__device__ double Log_Add(double a, double b) {
	if (!isfinite(a) && !isfinite(b)) {
		return __longlong_as_double(0xfff0000000000000ULL);
	}
	if (!isfinite(a)) {
		return b;
	}
	if (!isfinite(b)) {
		return a;
	}

	double max = (a > b) ? (a) : (b);

	return (max + log1p(exp(a + b - 2 * max)));
}
__device__ void Backward_Algorithm(int number_labels, int length_event, int length_label_sequence, int maximum_length_label_sequence, int label_sequence[], float _likelihood[], double &log_likelihood, double beta[]) {
	// if (threadIdx.x == 0) log_likelihood = 0;

	for (int t = length_event - 1; t >= 0; t--) {
		int index = t * maximum_length_label_sequence;

		float *likelihood = &_likelihood[t * number_labels];

		double sum = __longlong_as_double(0xfff0000000000000ULL);

		if (t == length_event - 1) {
			for (int s = threadIdx.x; s < length_label_sequence; s += blockDim.x) {
				beta[index + s] = log((s >= length_label_sequence - 2) * likelihood[label_sequence[s]]);
			}
		}
		else {
			int next_index = (t + 1) * maximum_length_label_sequence;

			for (int s = threadIdx.x; s < length_label_sequence; s += blockDim.x) {
				double sum = __longlong_as_double(0xfff0000000000000ULL);

				if (s <= 2 * t + 1) {
					if (label_sequence[s] == number_labels - 1 || (s <= length_label_sequence - 3 && label_sequence[s + 2] == label_sequence[s])) {
						sum = (s == length_label_sequence - 1) ? (beta[next_index + s]) : (Log_Add(beta[next_index + s], beta[next_index + s + 1]));
					}
					else {
						sum = (s == length_label_sequence - 2) ? (Log_Add(beta[next_index + s], beta[next_index + s + 1])) : (Log_Add(Log_Add(beta[next_index + s], beta[next_index + s + 1]), beta[next_index + s + 2]));
					}
				}
				beta[index + s] = sum + log(likelihood[label_sequence[s]]);
			}
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			for (int s = 0; s < length_label_sequence; s++) {
				sum = Log_Add(sum, beta[index + s]);
			}
			for (int s = 0; s < length_label_sequence; s++) {
				beta[index + s] -= sum;
			}
			// log_likelihood += sum;
		}
		__syncthreads();
	}
}
__device__ void Forward_Algorithm(int number_labels, int length_event, int length_label_sequence, int maximum_length_label_sequence, int label_sequence[], float _likelihood[], double &log_likelihood, double alpha[]) {
	if (threadIdx.x == 0) log_likelihood = 0;

	for (int t = 0; t < length_event; t++) {
		int index = t * maximum_length_label_sequence;

		float *likelihood = &_likelihood[t * number_labels];

		double sum = __longlong_as_double(0xfff0000000000000ULL);

		if (t == 0) {
			for (int s = threadIdx.x; s < length_label_sequence; s += blockDim.x) {
				alpha[index + s] = log((s <= 1) * likelihood[label_sequence[s]]);
			}
		}
		else {
			int previous_index = (t - 1) * maximum_length_label_sequence;

			for (int s = threadIdx.x; s < length_label_sequence; s += blockDim.x) {
				double sum = __longlong_as_double(0xfff0000000000000ULL);

				if (s >= (length_label_sequence - 1) - 2 * ((length_event - 1) - t) - 1) {
					if (label_sequence[s] == number_labels - 1 || (s >= 2 && label_sequence[s - 2] == label_sequence[s])) {
						sum = (s == 0) ? (alpha[previous_index + s]) : (Log_Add(alpha[previous_index + s], alpha[previous_index + s - 1]));
					}
					else {
						sum = (s == 1) ? (Log_Add(alpha[previous_index + s], alpha[previous_index + s - 1])) : (Log_Add(Log_Add(alpha[previous_index + s], alpha[previous_index + s - 1]), alpha[previous_index + s - 2]));
					}
				}
				alpha[index + s] = sum + log(likelihood[label_sequence[s]]);
			}
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			for (int s = 0; s < length_label_sequence; s++) {
				sum = Log_Add(sum, alpha[index + s]);
			}
			for (int s = 0; s < length_label_sequence; s++) {
				alpha[index + s] -= sum;
			}
			log_likelihood += sum;
		}
		__syncthreads();
	}
}

__global__ void Calculate_Error(int maximum_length_label_sequence, int number_labels, int time_step, int length_event[], int length_label_sequence[], int _label_sequence[], float _error[], float _likelihood[], double _alpha[], double _beta[], double log_likelihood[]) {
	int h = blockIdx.x;

	int *label_sequence = &_label_sequence[h * maximum_length_label_sequence];

	float *error = &_error[h * time_step * number_labels];
	float *likelihood = &_likelihood[h * time_step * number_labels];

	double *alpha = &_alpha[h * time_step * maximum_length_label_sequence];
	double *beta = &_beta[h * time_step * maximum_length_label_sequence];

	Forward_Algorithm(number_labels, length_event[h], length_label_sequence[h], maximum_length_label_sequence, label_sequence, likelihood, log_likelihood[h], alpha);
	Backward_Algorithm(number_labels, length_event[h], length_label_sequence[h], maximum_length_label_sequence, label_sequence, likelihood, log_likelihood[h], beta);

	for (int s = threadIdx.x; s < length_event[h] * number_labels; s += blockDim.x) {
		int t = s / number_labels;
		int i = s % number_labels;

		int index[2] = { t * number_labels, t * maximum_length_label_sequence };

		double sum[2] = { __longlong_as_double(0xfff0000000000000ULL), __longlong_as_double(0xfff0000000000000ULL) };

		for (int j = 0; j < length_label_sequence[h]; j++) {
			int k = label_sequence[j];

			if (i == k) {
				sum[1] = Log_Add(sum[1], alpha[index[1] + j] + beta[index[1] + j]);
			}
			sum[0] = Log_Add(sum[0], alpha[index[1] + j] + beta[index[1] + j] - log(likelihood[index[0] + k]));
		}
		error[index[0] + i] = likelihood[index[0] + i] - exp(sum[1] - log(likelihood[index[0] + i]) - sum[0]);
	}
}


Batch_Normalization::Batch_Normalization(int number_maps, int map_size) {
	this->map_size = map_size;
	this->number_maps = number_maps;
	batch_size = 1;
	number_nodes = number_maps * map_size;
	time_step = 1;

	cudaMalloc(&gamma, sizeof(float) * number_maps);
	gamma_optimizer = new Optimizer();
	gamma_optimizer->Resize_Memory(number_maps);

	cudaMalloc(&beta, sizeof(float) * number_maps);
	beta_optimizer = new Optimizer();
	beta_optimizer->Resize_Memory(number_maps);

	cudaMalloc(&mean, sizeof(float) * number_maps);
	cudaMalloc(&variance, sizeof(float) * number_maps);
	cudaMalloc(&sum_mean, sizeof(float) * number_maps);
	cudaMemset(sum_mean, 0, sizeof(float) * number_maps);
	cudaMalloc(&sum_variance, sizeof(float) * number_maps);
	cudaMemset(sum_variance, 0, sizeof(float) * number_maps);

	cudaMalloc(&error_backup, sizeof(float) * number_nodes);
	cudaMalloc(&error_normalized, sizeof(float) * number_nodes);
	cudaMalloc(&neuron_backup, sizeof(float) * number_nodes);
	cudaMalloc(&neuron_normalized, sizeof(float) * number_nodes);
}
Batch_Normalization::~Batch_Normalization() {}

void Batch_Normalization::Activate(string phase, float neuron[], int time_index) {
	::Activate << <number_maps, NUMBER_THREADS >> > (phase == "training", time_index, neuron, *this);
}
void Batch_Normalization::Adjust_Parameter(double gradient_clip, double learning_rate) {
	::Adjust_Parameter << < number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (number_maps, gamma, gradient_clip, learning_rate, *gamma_optimizer);
	::Adjust_Parameter << < number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (number_maps, beta, gradient_clip, learning_rate, *beta_optimizer);
}
void Batch_Normalization::Calculate_Mean_Variance(int number_batches) {
	Multiply << <time_step * number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (time_step * number_maps, sum_mean, 1.0 / number_batches, mean);
	Multiply << <time_step * number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (time_step * number_maps, sum_variance, batch_size / ((batch_size - 1.0) * number_batches), variance);
	cudaMemset(sum_mean, 0, sizeof(float) * time_step * number_maps);
	cudaMemset(sum_variance, 0, sizeof(float) * time_step * number_maps);
}
void Batch_Normalization::Destroy() {
	cudaFree(gamma);
	cudaFree(beta);
	cudaFree(mean);
	cudaFree(variance);
	cudaFree(sum_mean);
	cudaFree(sum_variance);

	cudaFree(error_backup);
	cudaFree(error_normalized);
	cudaFree(neuron_backup);
	cudaFree(neuron_normalized);

	gamma_optimizer->Destroy();
	delete gamma_optimizer;
	beta_optimizer->Destroy();
	delete beta_optimizer;
}
void Batch_Normalization::Differentiate(float error[], int time_index) {
	::Differentiate << <number_maps, NUMBER_THREADS >> > (time_index, error, *this);
}
void Batch_Normalization::Initialize(double gamma) {
	Set << <number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (number_maps, this->gamma, gamma);
	cudaMemset(beta, 0, sizeof(float) * number_maps);
	cudaMemset(sum_mean, 0, sizeof(float) * time_step * number_maps);
	cudaMemset(sum_variance, 0, sizeof(float) * time_step * number_maps);
}
void Batch_Normalization::Load(ifstream &file) {
	float *memory = new float[time_step * number_maps];

	for (int j = 0; j < number_maps; j++) file >> memory[j];
	cudaMemcpy(gamma, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < number_maps; j++) file >> memory[j];
	cudaMemcpy(beta, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < time_step * number_maps; j++) file >> memory[j];
	cudaMemcpy(mean, memory, sizeof(float) * time_step * number_maps, cudaMemcpyHostToDevice);
	for (int j = 0; j < time_step * number_maps; j++) file >> memory[j];
	cudaMemcpy(variance, memory, sizeof(float) * time_step * number_maps, cudaMemcpyHostToDevice);

	delete[] memory;
}
void Batch_Normalization::Resize_Memory(int batch_size, int time_step) {
	if (this->batch_size != batch_size || this->time_step != time_step) {
		int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

		if (this->time_step != time_step) {
			int memory_size = sizeof(float) * time_step * number_maps;

			cudaFree(mean);
			cudaFree(variance);
			cudaFree(sum_mean);
			cudaFree(sum_variance);

			cudaMalloc(&mean, memory_size);
			cudaMalloc(&variance, memory_size);
			cudaMalloc(&sum_mean, memory_size);
			cudaMemset(sum_mean, 0, memory_size);
			cudaMalloc(&sum_variance, memory_size);
			cudaMemset(sum_variance, 0, memory_size);
		}
		cudaFree(error_backup);
		cudaFree(error_normalized);
		cudaFree(neuron_backup);
		cudaFree(neuron_normalized);

		cudaMalloc(&error_backup, memory_size);
		cudaMalloc(&error_normalized, memory_size);
		cudaMalloc(&neuron_backup, memory_size);
		cudaMalloc(&neuron_normalized, memory_size);

		this->batch_size = batch_size;
		this->time_step = time_step;
	}
}
void Batch_Normalization::Save(ofstream &file) {
	float *memory = new float[time_step * number_maps];

	cudaMemcpy(memory, gamma, sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, beta, sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, mean, sizeof(float) * time_step * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < time_step * number_maps; j++) file << memory[j] << endl;
	cudaMemcpy(memory, variance, sizeof(float) * time_step * number_maps, cudaMemcpyDeviceToHost);
	for (int j = 0; j < time_step * number_maps; j++) file << memory[j] << endl;

	delete[] memory;
}
void Batch_Normalization::Set_Optimizer(Optimizer *optimizer) {
	gamma_optimizer->Destroy();
	delete gamma_optimizer;
	gamma_optimizer = optimizer->Copy(number_maps);

	beta_optimizer->Destroy();
	delete beta_optimizer;
	beta_optimizer = optimizer->Copy(number_maps);
}

double Batch_Normalization::Calculate_Gradient(double learning_rate) {
	float sum_gradient = 0, *gradient;

	cudaMalloc(&gradient, sizeof(float) * number_maps);

	::Calculate_Gradient << <number_maps, NUMBER_THREADS >> > (gradient, learning_rate, *this, *gamma_optimizer, *beta_optimizer);
	Merge << <1, NUMBER_THREADS >> > (number_maps, gradient);
	cudaMemcpy(&sum_gradient, gradient, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(gradient);

	return sum_gradient;
}


Connection::Connection(string properties) {
	this->properties = properties;

	from_errors = nullptr;
	from_neurons = nullptr;
	from_weights = nullptr;
	optimizer = nullptr;
	weight = nullptr;
	LSTM_weight = nullptr;
}
Connection::~Connection() {}

void Connection::Destroy() {
	if (from_errors) {
		cudaFree(from_errors);
	}
	if (from_neurons) {
		cudaFree(from_neurons);
	}
	if (from_weights) {
		cudaFree(from_weights);
	}
	if (optimizer) {
		optimizer->Destroy();
		delete optimizer;
	}
	if (weight) {
		cudaFree(weight);
	}
	if (LSTM_weight) {
		delete LSTM_weight;
	}
}
void Connection::Initialize(double scale) {
	if (weight) {
		float *memory = new float[number_weights];

		for (int j = 0; j < number_weights; j++) {
			memory[j] = scale * (2.0 * rand() / RAND_MAX - 1);
		}
		cudaMemcpy(weight, memory, sizeof(float) * number_weights, cudaMemcpyHostToDevice);
		delete[] memory;
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
			float *memory = new float[number_weights];

			for (int j = 0; j < number_weights; j++) {
				memory[j] = scale * (2.0 * rand() / RAND_MAX - 1);
			}
			cudaMemcpy(LSTM_weight->weight[h], memory, sizeof(float) * number_weights, cudaMemcpyHostToDevice);
			delete[] memory;
		}
	}
}
void Connection::Load(ifstream &file) {
	if (weight || LSTM_weight) {
		float *memory = new float[number_weights];

		if (weight) {
			for (int j = 0; j < number_weights; j++) file >> memory[j];
			cudaMemcpy(weight, memory, sizeof(float) * number_weights, cudaMemcpyHostToDevice);
		}
		if (LSTM_weight) {
			for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
				for (int j = 0; j < number_weights; j++) file >> memory[j];
				cudaMemcpy(LSTM_weight->weight[h], memory, sizeof(float) * number_weights, cudaMemcpyHostToDevice);
			}
		}
		delete[] memory;
	}
}
void Connection::Load(string path) {
	ifstream file(path);

	if (file.is_open()) {
		Load(file);
		file.close();
	}
	else {
		cerr << "[Connection], " + path + " not found" << endl;
	}
}
void Connection::Save(ofstream &file) {
	if (weight || LSTM_weight) {
		float *memory = new float[number_weights];

		if (weight) {
			cudaMemcpy(memory, weight, sizeof(float) * number_weights, cudaMemcpyDeviceToHost);
			for (int j = 0; j < number_weights; j++) file << memory[j] << endl;
		}
		if (LSTM_weight) {
			for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
				cudaMemcpy(memory, LSTM_weight->weight[h], sizeof(float) * number_weights, cudaMemcpyDeviceToHost);
				for (int j = 0; j < number_weights; j++) file << memory[j] << endl;
			}
		}
	}
}
void Connection::Save(string path) {
	ofstream file(path);

	Save(file);
	file.close();
}
void Connection::Set_Optimizer(Optimizer *optimizer) {
	if (weight) {
		this->optimizer->Destroy();
		delete this->optimizer;
		this->optimizer = optimizer->Copy(number_weights);
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
			LSTM_weight->optimizer[h]->Destroy();
			delete LSTM_weight->optimizer[h];
			LSTM_weight->optimizer[h] = optimizer->Copy(number_weights);
		}
	}
}

int Connectionist_Temporal_Classification::Search_Label(string label) {
	auto l = label_index.find(label);

	if (l != label_index.end()) {
		return l->second;
	}
	cerr << "[Search_Label], label '" + label + "' not found" << endl;
	return -1;
}

double *Connectionist_Temporal_Classification::Get_Probability(string label, unordered_map<string, double> &probability) {
	auto p = probability.find(label);

	if (p == probability.end()) {
		probability.insert(pair<string, double>(label, 0));
		p = probability.find(label);
	}
	return &p->second;
}

Connectionist_Temporal_Classification::Connectionist_Temporal_Classification(int number_labels, string label[]) {
	this->number_labels = number_labels;
	this->label = new string[number_labels];

	for (int i = 0; i < number_labels; i++) {
		label_index.insert(pair<string, int>(this->label[i] = label[i], i));
	}
}
Connectionist_Temporal_Classification::~Connectionist_Temporal_Classification() {
	delete[] label;
}

void Connectionist_Temporal_Classification::Best_Path_Decoding(int length_event, float _likelihood[], vector<string> &label_sequence, bool space_between_labels) {
	string token;

	for (int t = 0, argmax, previous_state = number_labels - 1; t < length_event; t++) {
		float max;

		float *likelihood = &_likelihood[t * number_labels];

		for (int i = 0; i < number_labels; i++) {
			if (i == 0 || max < likelihood[i]) {
				max = likelihood[argmax = i];
			}
		}
		if (previous_state != argmax) {
			token += label[argmax];

			if ((space_between_labels && !token.empty()) || label[argmax] == " ") {
				label_sequence.push_back(token);
				token.clear();
			}
			previous_state = argmax;
		}
	}
}
void Connectionist_Temporal_Classification::Calculate_Error(vector<string> target_label_sequence[], int batch_size, int time_step, int _length_event[], float error[], float likelihood[], double _log_likelihood[]) {
	int maximum_length_label_sequence = 0;

	int *label_sequence;
	int *length_event;
	int *length_label_sequence;

	double *alpha;
	double *beta;
	double *log_likelihood;

	for (int h = 0; h < batch_size; h++) {
		if (maximum_length_label_sequence < static_cast<int>(target_label_sequence[h].size())) {
			maximum_length_label_sequence = static_cast<int>(target_label_sequence[h].size());
		}
	}
	cudaMalloc(&label_sequence, sizeof(int) * batch_size * maximum_length_label_sequence);
	cudaMalloc(&length_event, sizeof(int) * batch_size);
	cudaMemcpy(length_event, _length_event, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
	cudaMalloc(&length_label_sequence, sizeof(int) * batch_size);

	for (int h = 0; h < batch_size; h++) {
		int *label_index = new int[target_label_sequence[h].size()], length_label = static_cast<int>(target_label_sequence[h].size());

		for (int t = 0; t < target_label_sequence[h].size(); t++) {
			label_index[t] = Search_Label(target_label_sequence[h][t]);
		}
		cudaMemcpy(&label_sequence[h * maximum_length_label_sequence], label_index, sizeof(int) * target_label_sequence[h].size(), cudaMemcpyHostToDevice);
		cudaMemcpy(&length_label_sequence[h], &length_label, sizeof(int), cudaMemcpyHostToDevice);

		delete[] label_index;
	}
	cudaMalloc(&alpha, sizeof(double) * batch_size * time_step * maximum_length_label_sequence);
	cudaMalloc(&beta, sizeof(double) * batch_size * time_step * maximum_length_label_sequence);
	cudaMalloc(&log_likelihood, sizeof(double) * batch_size);

	::Calculate_Error << <batch_size, NUMBER_THREADS >> > (maximum_length_label_sequence, number_labels, time_step, length_event, length_label_sequence, label_sequence, error, likelihood, alpha, beta, log_likelihood);

	cudaMemcpy(_log_likelihood, log_likelihood, sizeof(double) * batch_size, cudaMemcpyDeviceToHost);

	cudaFree(alpha);
	cudaFree(beta);
	cudaFree(label_sequence);
	cudaFree(length_event);
	cudaFree(length_label_sequence);
	cudaFree(log_likelihood);
}

bool comparator(const pair<double, string> &a, const pair<double, string> &b) {
	return a.first > b.first;
}

void Connectionist_Temporal_Classification::Prefix_Beam_Search_Decoding(int length_event, float _likelihood[], vector<string> &label_sequence, int k, bool space_between_labels) {
	set<string> A_prev = { "" };

	unordered_map<string, double> *Pb = new unordered_map<string, double>[length_event + 1];
	unordered_map<string, double> *Pnb = new unordered_map<string, double>[length_event + 1];

	Pb[0].insert(pair<string, double>("", 1));
	Pnb[0].insert(pair<string, double>("", 0));

	for (int t = 1; t <= length_event; t++) {
		float *likelihood = &_likelihood[(t - 1) * number_labels];

		set<string> A_next;

		vector<pair<double, string>> v;

		for (auto l = A_prev.begin(); l != A_prev.end(); l++) {
			for (int c = 0; c < number_labels; c++) {
				if (likelihood[c] > 0.00000001) {
					if (label[c] == "") {
						*Get_Probability(*l, Pb[t]) += likelihood[c] * (*Get_Probability(*l, Pb[t - 1]) + *Get_Probability(*l, Pnb[t - 1]));
						A_next.insert(*l);
					}
					else {
						int index = static_cast<int>((*l).size() - label[c].size());

						string l_plus = ((*l).empty()) ? (label[c]) : ((space_between_labels) ? (*l + " " + label[c]) : (*l + label[c]));

						if (index >= 0 && &(*l)[index] == label[c]) {
							*Get_Probability(l_plus, Pnb[t]) += likelihood[c] * *Get_Probability(*l, Pb[t - 1]);
							*Get_Probability(*l, Pnb[t]) += likelihood[c] * *Get_Probability(*l, Pnb[t - 1]);
						}
						else {
							*Get_Probability(l_plus, Pnb[t]) += likelihood[c] * (*Get_Probability(*l, Pb[t - 1]) + *Get_Probability(*l, Pnb[t - 1]));
						}
						if (A_prev.find(l_plus) == A_prev.end()) {
							*Get_Probability(l_plus, Pb[t]) += likelihood[number_labels - 1] * (*Get_Probability(l_plus, Pb[t - 1]) + *Get_Probability(l_plus, Pnb[t - 1]));
							*Get_Probability(l_plus, Pnb[t]) += likelihood[c] * *Get_Probability(l_plus, Pnb[t - 1]);
						}
						A_next.insert(l_plus);
					}
				}
			}
		}
		A_prev.clear();

		for (auto a = A_next.begin(); a != A_next.end(); a++) {
			v.push_back(pair<double, string>(*Get_Probability(*a, Pb[t]) + *Get_Probability(*a, Pnb[t]), *a));
		}
		sort(v.begin(), v.end(), comparator);

		for (int i = 0; i < k && i < static_cast<int>(v.size()); i++) {
			A_prev.insert(v[i].second);
		}
	}
	delete[] Pb;
	delete[] Pnb;

	istringstream iss(*A_prev.begin());

	for (string s; getline(iss, s, ' ');) {
		label_sequence.push_back(s);
	}
}

Layer::Layer(string properties, int number_maps, int map_width, int map_height, int map_depth) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::~Layer() {}

void Layer::Construct() {
	bool batch_normalization = (strstr(properties.c_str(), "BN") != 0);

	batch_size = 0;
	map_size = map_depth * map_height * map_width;
	number_connections = 0;
	number_nodes = number_maps * map_size;
	time_step = 0;

	this->batch_normalization[0] = nullptr;
	this->batch_normalization[1] = nullptr;

	error[0] = nullptr;
	error[1] = nullptr;
	neuron[0] = nullptr;
	neuron[1] = nullptr;
	neuron[2] = nullptr;

	bias = nullptr;
	bias_optimizer = nullptr;
	dropout_mask = nullptr;
	slope = nullptr;
	slope_optimizer = nullptr;
	time_mask = nullptr;
	time_mask_device = nullptr;
	LSTM_node = nullptr;

	if (strstr(properties.c_str(), "dropout")) {
		cudaMalloc(&dropout_mask, number_maps);
	}
	if (strstr(properties.c_str(), "LSTM")) {
		cudaMalloc(&error[0], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[0], sizeof(float) * number_nodes);
		LSTM_node = new LSTM_Node(number_maps, map_size, batch_normalization);
	}
	else if (strstr(properties.c_str(), "RNN")) {
		if (batch_normalization) {
			this->batch_normalization[0] = new Batch_Normalization(number_maps, map_size);
			this->batch_normalization[1] = new Batch_Normalization(number_maps, map_size);
		}
		cudaMalloc(&error[0], sizeof(float) * number_nodes);
		cudaMalloc(&error[1], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[0], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[1], sizeof(float) * number_nodes);
	}
	else {
		if (batch_normalization) {
			this->batch_normalization[0] = new Batch_Normalization(number_maps, map_size);
		}
		if (strstr(properties.c_str(), "ELU")) {
			float parameter = atof(strstr(properties.c_str(), "ELU") + 3);

			cudaMalloc(&slope, sizeof(float));
			cudaMemcpy(slope, &parameter, sizeof(float), cudaMemcpyHostToDevice);
		}
		else if (strstr(properties.c_str(), "PReLU")) {
			float slope = atof(strstr(properties.c_str(), "PReLU") + 5);

			cudaMalloc(&neuron[2], sizeof(float) * number_nodes);
			cudaMalloc(&this->slope, sizeof(float) * number_nodes);
			Set << <number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (number_nodes, this->slope, slope);

			slope_optimizer = new Optimizer();
			slope_optimizer->Resize_Memory(number_nodes);
		}
		else if (strstr(properties.c_str(), "ReLU")) {
			float slope = atof(strstr(properties.c_str(), "ReLU") + 4);

			cudaMalloc(&this->slope, sizeof(float));
			cudaMemcpy(this->slope, &slope, sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMalloc(&error[0], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[0], sizeof(float) * number_nodes);
	}
}
void Layer::Destroy() {
	if (batch_normalization[0]) {
		batch_normalization[0]->Destroy();
		delete batch_normalization[0];
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Destroy();
		delete batch_normalization[1];
	}
	if (error[0]) {
		cudaFree(error[0]);
	}
	if (error[1]) {
		cudaFree(error[1]);
	}
	if (neuron[0]) {
		cudaFree(neuron[0]);
	}
	if (neuron[1]) {
		cudaFree(neuron[1]);
	}
	if (neuron[2]) {
		cudaFree(neuron[2]);
	}
	if (bias) {
		cudaFree(bias);
	}
	if (bias_optimizer) {
		bias_optimizer->Destroy();
		delete bias_optimizer;
	}
	if (slope_optimizer) {
		cudaFree(slope);
		slope_optimizer->Destroy();
		delete slope_optimizer;
	}
	if (time_mask) {
		cudaFree(time_mask_device);
		delete[] time_mask;
	}
	if (LSTM_node) {
		delete LSTM_node;
	}
}
void Layer::Disconnect(Layer *target_layer) {
	if (target_layer == nullptr) {
		cerr << "[Disconnect], target_layer = nullptr" << endl;
		return;
	}
	for (int i = 0; i < number_connections; i++) {
		if (parent_layer[i] == target_layer) {
			connection.erase(connection.begin() + i);
			parent_layer.erase(parent_layer.begin() + i);
			number_connections = static_cast<int>(connection.size());
			break;
		}
	}
}
void Layer::Initialize(double scale, double gamma) {
	for (int h = 0; h < connection.size(); h++) {
		connection[h]->Initialize(scale);
	}
	if (batch_normalization[0]) {
		batch_normalization[0]->Initialize(gamma);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Initialize(gamma);
	}
	if (bias) {
		float *memory = new float[number_maps];

		for (int j = 0; j < number_maps; j++) {
			memory[j] = scale * (2.0 * rand() / RAND_MAX - 1);
		}
		cudaMemcpy(bias, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
		delete[] memory;
	}
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Initialize(gamma);
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Initialize(gamma);
		}
		for (int h = 0; h < LSTM_node->number_weight_types; h++) {
			float *memory = new float[number_maps];

			for (int j = 0; j < number_maps; j++) {
				memory[j] = scale * (2.0 * rand() / RAND_MAX - 1);
			}
			cudaMemcpy(LSTM_node->bias[h], memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
			delete[] memory;
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			float *memory = new float[number_maps];

			for (int j = 0; j < number_maps; j++) {
				memory[j] = scale * (2.0 * rand() / RAND_MAX - 1);
			}
			cudaMemcpy(LSTM_node->peephole[h], memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
			delete[] memory;
		}
	}
}
void Layer::Load(ifstream &file) {
	if (batch_normalization[0]) {
		batch_normalization[0]->Load(file);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Load(file);
	}
	if (bias || slope_optimizer || LSTM_node) {
		float *memory = new float[number_nodes];

		if (bias) {
			for (int j = 0; j < number_maps; j++) file >> memory[j];
			cudaMemcpy(bias, memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
		}
		if (slope_optimizer) {
			for (int j = 0; j < number_nodes; j++) file >> memory[j];
			cudaMemcpy(slope, memory, sizeof(float) * number_nodes, cudaMemcpyHostToDevice);
		}
		if (LSTM_node) {
			for (int h = 0; h < LSTM_node->number_node_types; h++) {
				if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Load(file);
				if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Load(file);
			}
			for (int h = 0; h < LSTM_node->number_weight_types; h++) {
				for (int j = 0; j < number_maps; j++) file >> memory[j];
				cudaMemcpy(LSTM_node->bias[h], memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
			}
			for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
				for (int j = 0; j < number_maps; j++) file >> memory[j];
				cudaMemcpy(LSTM_node->peephole[h], memory, sizeof(float) * number_maps, cudaMemcpyHostToDevice);
			}
		}
		delete[] memory;
	}
}
void Layer::Load(string path) {
	ifstream file(path);

	if (file.is_open()) {
		Load(file);
		file.close();
	}
	else {
		cerr << "[Layer], " + path + " not found" << endl;
	}
}
void Layer::Resize_Memory(int batch_size, int time_step) {
	if (this->batch_size != batch_size || this->time_step != time_step) {
		int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

		if (batch_normalization[0]) {
			batch_normalization[0]->Resize_Memory(batch_size, time_step);
		}
		if (batch_normalization[1]) {
			batch_normalization[1]->Resize_Memory(batch_size, time_step);
		}
		if (dropout_mask) {
			cudaFree(dropout_mask);
			cudaMalloc(&dropout_mask, batch_size * number_maps);
		}
		if (error[0]) {
			cudaFree(error[0]);
			cudaMalloc(&error[0], memory_size);
		}
		if (error[1]) {
			cudaFree(error[1]);
			cudaMalloc(&error[1], memory_size);
		}
		if (neuron[0]) {
			cudaFree(neuron[0]);
			cudaMalloc(&neuron[0], memory_size);
		}
		if (neuron[1]) {
			cudaFree(neuron[1]);
			cudaMalloc(&neuron[1], memory_size);
		}
		if (neuron[2]) {
			cudaFree(neuron[2]);
			cudaMalloc(&neuron[2], memory_size);
		}
		if (LSTM_node) {
			LSTM_node->Resize_Memory(batch_size, time_step);
		}
		this->batch_size = batch_size;
		this->time_step = time_step;
	}
}
void Layer::Save(ofstream &file) {
	if (batch_normalization[0]) {
		batch_normalization[0]->Save(file);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Save(file);
	}
	if (bias || slope_optimizer || LSTM_node) {
		float *memory = new float[number_nodes];

		if (bias) {
			cudaMemcpy(memory, bias, sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
			for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
		}
		if (slope_optimizer) {
			cudaMemcpy(memory, slope, sizeof(float) * number_nodes, cudaMemcpyDeviceToHost);
			for (int j = 0; j < number_nodes; j++) file << memory[j] << endl;
		}
		if (LSTM_node) {
			for (int h = 0; h < LSTM_node->number_node_types; h++) {
				if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Save(file);
				if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Save(file);
			}
			for (int h = 0; h < LSTM_node->number_weight_types; h++) {
				cudaMemcpy(memory, LSTM_node->bias[h], sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
				for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
			}
			for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
				cudaMemcpy(memory, LSTM_node->peephole[h], sizeof(float) * number_maps, cudaMemcpyDeviceToHost);
				for (int j = 0; j < number_maps; j++) file << memory[j] << endl;
			}
		}
		delete[] memory;
	}
}
void Layer::Save(string path) {
	ofstream file(path);

	Save(file);
	file.close();
}
void Layer::Set_Epsilon(double epsilon) {
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->epsilon = epsilon;
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->epsilon = epsilon;
		}
	}
	if (batch_normalization[0]) batch_normalization[0]->epsilon = epsilon;
	if (batch_normalization[1]) batch_normalization[1]->epsilon = epsilon;
}
void Layer::Set_Optimizer(Optimizer *optimizer) {
	if (optimizer == nullptr) {
		cerr << "[Set_Optimizer], optimizer = nullptr" << endl;
		return;
	}
	if (bias_optimizer) {
		bias_optimizer->Destroy();
		delete bias_optimizer;
		bias_optimizer = optimizer->Copy(number_maps);
	}
	if (slope_optimizer) {
		slope_optimizer->Destroy();
		delete slope_optimizer;
		slope_optimizer = optimizer->Copy(number_nodes);
	}

	for (int h = 0; h < connection.size(); h++) {
		connection[h]->Set_Optimizer(optimizer);
	}
	if (batch_normalization[0]) {
		batch_normalization[0]->Set_Optimizer(optimizer);
	}
	if (batch_normalization[1]) {
		batch_normalization[1]->Set_Optimizer(optimizer);
	}
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Set_Optimizer(optimizer);
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Set_Optimizer(optimizer);
		}
		for (int h = 0; h < LSTM_node->number_weight_types; h++) {
			LSTM_node->bias_optimizer[h]->Destroy();
			delete LSTM_node->bias_optimizer[h];
			LSTM_node->bias_optimizer[h] = optimizer->Copy(number_maps);
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			LSTM_node->peephole_optimizer[h]->Destroy();
			delete LSTM_node->peephole_optimizer[h];
			LSTM_node->peephole_optimizer[h] = optimizer->Copy(number_maps);
		}
	}
	delete optimizer;
}
void Layer::Set_Time_Mask(bool time_mask[]) {
	if (this->time_mask) {
		cudaFree(time_mask_device);
		delete[] this->time_mask;
	}
	this->time_mask = time_mask;
	cudaMalloc(&time_mask_device, time_step);
	cudaMemcpy(time_mask_device, time_mask, time_step, cudaMemcpyHostToDevice);
}

bool Layer::Check_Mask(int time_index) {
	return (time_mask == nullptr || time_mask[time_index]);
}

Connection* Layer::Connect(Layer *parent_layer, string properties) {
	if (parent_layer == nullptr) {
		cerr << "[Connect], parent_layer = nullptr" << endl;
		return nullptr;
	}
	if (properties.empty()) {
		cerr << "[Connect], properties is empty" << endl;
		return nullptr;
	}
	if (strstr(properties.c_str(), "add") && number_nodes != parent_layer->number_nodes) {
		cerr << "[Connect], add connection requires: (number_nodes = parent_layer->number_nodes)" << endl;
		return nullptr;
	}
	if (strstr(properties.c_str(), "dilate")) {
		if (map_depth < parent_layer->map_depth || map_height < parent_layer->map_height || map_width < parent_layer->map_width) {
			cerr << "[Connect], dilated convolution requires: (map_size >= parent_layer->map_size)" << endl;
			return nullptr;
		}
		if (number_maps != parent_layer->number_maps) {
			cerr << "[Connect], dilated convolution requires: (number_maps = parent_layer->number_maps)" << endl;
			return nullptr;
		}
	}
	if (strstr(properties.c_str(), "recurrent") && !(strstr(this->properties.c_str(), "RNN") || strstr(this->properties.c_str(), "LSTM"))) {
		cerr << "[Connect], recurrent connection is only available from the recurrent layer (RNN/LSTM)" << endl;
		return nullptr;
	}
	if (properties[0] == 'P' && number_maps != parent_layer->number_maps) {
		cerr << "[Connect], pooling layer requires: (number_maps = parent_layer->number_maps)" << endl;
		return nullptr;
	}

	unordered_map<int, int> weight_index;

	Connection *connection = new Connection(properties);

	this->connection.push_back(connection);
	number_connections = static_cast<int>(this->connection.size());
	this->parent_layer.push_back(parent_layer);

	// Set kernel size if specified
	if (const char *kernel_size = strstr(properties.c_str(), "kernel")) {
		const char *end = strstr(kernel_size, ")");

		connection->kernel_width = atoi(kernel_size + 7);
		kernel_size = strstr(kernel_size, "x");

		if (kernel_size && kernel_size < end && atoi(kernel_size + 1) > 0) {
			connection->kernel_height = atoi(kernel_size + 1);
			kernel_size = strstr(kernel_size + 1, "x");

			if (kernel_size && kernel_size < end && atoi(kernel_size + 1) > 0) {
				connection->kernel_depth = atoi(kernel_size + 1);
			}
			else {
				connection->kernel_depth = 1;
			}
		}
		else {
			connection->kernel_height = 1;
			connection->kernel_depth = 1;
		}
	}
	else if (properties[0] == 'P') {
		connection->kernel_width = 0;
		connection->kernel_height = 0;
		connection->kernel_depth = 0;
	}
	else {
		connection->kernel_width = abs(parent_layer->map_width - map_width) + 1;
		connection->kernel_height = abs(parent_layer->map_height - map_height) + 1;
		connection->kernel_depth = abs(parent_layer->map_depth - map_depth) + 1;
	}
	connection->kernel_size = connection->kernel_depth * connection->kernel_height * connection->kernel_width;

	// Set stride size if specified
	if (const char *stride_size = strstr(properties.c_str(), "stride")) {
		const char *end = strstr(stride_size, ")");

		connection->stride_width = atoi(stride_size + 7);
		stride_size = strstr(stride_size, "x");

		if (stride_size && stride_size < end && atoi(stride_size + 1) > 0) {
			connection->stride_height = atoi(stride_size + 1);
			stride_size = strstr(stride_size + 1, "x");

			if (stride_size && stride_size < end && atoi(stride_size + 1) > 0) {
				connection->stride_depth = atoi(stride_size + 1);
			}
			else {
				connection->stride_depth = 1;
			}
		}
		else {
			connection->stride_height = 1;
			connection->stride_depth = 1;
		}
	}
	else if (properties[0] == 'P') {
		connection->stride_width = (parent_layer->map_width > map_width) ? (parent_layer->map_width / map_width) : (map_width / parent_layer->map_width);
		connection->stride_height = (parent_layer->map_height > map_height) ? (parent_layer->map_height / map_height) : (map_height / parent_layer->map_height);
		connection->stride_depth = (parent_layer->map_depth > map_depth) ? (parent_layer->map_depth / map_depth) : (map_depth / parent_layer->map_depth);
	}
	else if (strstr(properties.c_str(), "dilate")) {
		connection->stride_width = (map_width == 1) ? (1) : ((map_width - 1) / (parent_layer->map_width - 1));
		connection->stride_height = (map_height == 1) ? (1) : ((map_height - 1) / (parent_layer->map_height - 1));
		connection->stride_depth = (map_depth == 1) ? (1) : ((map_depth - 1) / (parent_layer->map_depth - 1));
	}
	else {
		connection->stride_width = 1;
		connection->stride_height = 1;
		connection->stride_depth = 1;
	}

	// Allocate memory for weight
	if (properties[0] == 'W') {
		bool depthwise_separable = (strstr(properties.c_str(), "DS") != 0);

		connection->number_maps = number_maps;
		connection->number_nodes = number_nodes;
		connection->number_weights = 0;

		for (int j = 0, index = 0; j < number_maps; j++) {
			for (int k = 0; k < parent_layer->number_maps; k++) {
				if (!depthwise_separable || j == k) {
					for (int l = 0; l < connection->kernel_size; l++) {
						weight_index.insert(pair<int, int>(j * parent_layer->number_maps * connection->kernel_size + k * connection->kernel_size + l, index++));
					}
					connection->number_weights += connection->kernel_size;
				}
			}
		}

		if (strstr(this->properties.c_str(), "LSTM")) {
			connection->LSTM_weight = new LSTM_Weight(connection->number_weights);
		}
		else {
			connection->optimizer = new Optimizer();
			connection->optimizer->Resize_Memory(connection->number_weights);
			cudaMalloc(&connection->weight, sizeof(float) * connection->number_weights);

			if (bias == nullptr) {
				cudaMalloc(&bias, sizeof(float) * number_maps);
			}
			if (bias_optimizer == nullptr) {
				bias_optimizer = new Optimizer();
				bias_optimizer->Resize_Memory(number_maps);
			}
		}
	}

	if (properties[0] == 'P' || properties[0] == 'W') {
		bool depthwise_separable = (strstr(properties.c_str(), "DS") != 0);

		vector<Index> *from_error = new vector<Index>[parent_layer->number_nodes];
		vector<Index> *from_neuron = new vector<Index>[number_nodes];
		vector<Index> *from_weight = new vector<Index>[connection->number_weights];

		for (int j = 0; j < number_maps; j++) {
			for (int k = 0; k < map_depth; k++) {
				for (int l = 0; l < map_height; l++) {
					for (int m = 0; m < map_width; m++) {
						int node_index[2] = { j * map_size + k * map_height * map_width + l * map_width + m, };

						if (properties[0] == 'W') {
							for (int n = 0; n < parent_layer->number_maps; n++) {
								if (!depthwise_separable || j == n) {
									int distance[3];

									for (int o = 0; o < parent_layer->map_depth; o++) {
										distance[0] = (map_depth < parent_layer->map_depth) ? (o - k * connection->stride_depth) : (k - o * connection->stride_depth);
										if (0 <= distance[0] && distance[0] < connection->kernel_depth) {
											for (int p = 0; p < parent_layer->map_height; p++) {
												distance[1] = (map_height < parent_layer->map_height) ? (p - l * connection->stride_height) : (l - p * connection->stride_height);
												if (0 <= distance[1] && distance[1] < connection->kernel_height) {
													for (int q = 0; q < parent_layer->map_width; q++) {
														distance[2] = (map_width < parent_layer->map_width) ? (q - m * connection->stride_width) : (m - q * connection->stride_width);
														if (0 <= distance[2] && distance[2] < connection->kernel_width) {
															Index index;

															node_index[1] = n * parent_layer->map_size + o * parent_layer->map_height * parent_layer->map_width + p * parent_layer->map_width + q;

															index.prev_node = node_index[1];
															index.next_node = node_index[0];
															index.weight = weight_index.find(j * parent_layer->number_maps * connection->kernel_size + n * connection->kernel_size + distance[0] * connection->kernel_height * connection->kernel_width + distance[1] * connection->kernel_width + distance[2])->second;

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
								distance[0] = (map_depth < parent_layer->map_depth) ? (o - k * connection->stride_depth) : (k - o * connection->stride_depth);
								if (0 <= distance[0] && distance[0] < ((connection->kernel_depth) ? (connection->kernel_depth) : (connection->stride_depth))) {
									for (int p = 0; p < parent_layer->map_height; p++) {
										distance[1] = (map_height < parent_layer->map_height) ? (p - l * connection->stride_height) : (l - p * connection->stride_height);
										if (0 <= distance[1] && distance[1] < ((connection->kernel_height) ? (connection->kernel_height) : (connection->stride_height))) {
											for (int q = 0; q < parent_layer->map_width; q++) {
												distance[2] = (map_width < parent_layer->map_width) ? (q - m * connection->stride_width) : (m - q * connection->stride_width);
												if (0 <= distance[2] && distance[2] < ((connection->kernel_width) ? (connection->kernel_width) : (connection->stride_width))) {
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
						else if (strstr(properties.c_str(), "dilate")) {
							for (int o = 0; o < parent_layer->map_depth; o++) {
								if (k - o * connection->stride_depth == 0) {
									for (int p = 0; p < parent_layer->map_height; p++) {
										if (l - p * connection->stride_height == 0) {
											for (int q = 0; q < parent_layer->map_width; q++) {
												if (m - q * connection->stride_width == 0) {
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

		int max = 0;

		Index index, *temp = new Index[0];

		index.weight = -1;

		for (int j = 0; j < parent_layer->number_nodes; j++) {
			from_error[j].push_back(index);

			if (max < from_error[j].size()) {
				max = static_cast<int>(from_error[j].size());
			}
		}
		cudaMallocManaged(&connection->from_errors, sizeof(Index) * max * parent_layer->number_nodes);
		temp = (Index*)realloc(temp, sizeof(Index) * max * parent_layer->number_nodes);

		for (int j = 0; j < parent_layer->number_nodes; j++) {
			for (int k = 0; k < from_error[j].size(); k++) {
				temp[j + k * parent_layer->number_nodes] = from_error[j][k];
			}
		}
		cudaMemcpy(connection->from_errors, temp, sizeof(Index) * max * parent_layer->number_nodes, cudaMemcpyHostToDevice);

		max = 0;
		for (int j = 0; j < number_nodes; j++) {
			from_neuron[j].push_back(index);

			if (max < from_neuron[j].size()) {
				max = static_cast<int>(from_neuron[j].size());
			}
		}
		cudaMallocManaged(&connection->from_neurons, sizeof(Index) * max * number_nodes);
		temp = (Index*)realloc(temp, sizeof(Index) * max * number_nodes);

		for (int j = 0; j < number_nodes; j++) {
			for (int k = 0; k < from_neuron[j].size(); k++) {
				temp[j + k * number_nodes] = from_neuron[j][k];
			}
		}
		cudaMemcpy(connection->from_neurons, temp, sizeof(Index) * max * number_nodes, cudaMemcpyHostToDevice);

		if (connection->number_weights) {
			int max = 0;

			for (int j = 0; j < connection->number_weights; j++) {
				from_weight[j].push_back(index);

				if (max < from_weight[j].size()) {
					max = static_cast<int>(from_weight[j].size());
				}
			}
			cudaMallocManaged(&connection->from_weights, sizeof(Index) * max * connection->number_weights);
			temp = (Index*)realloc(temp, sizeof(Index) * max * connection->number_weights);

			for (int j = 0; j < connection->number_weights; j++) {
				for (int k = 0; k < from_weight[j].size(); k++) {
					temp[j + k * connection->number_weights] = from_weight[j][k];
				}
			}
			cudaMemcpy(connection->from_weights, temp, sizeof(Index) * max * connection->number_weights, cudaMemcpyHostToDevice);
		}
		else {
			connection->from_weights = nullptr;
		}

		delete[] from_error;
		delete[] from_neuron;
		delete[] from_weight;
		delete[] temp;
	}
	return connection;
}


LSTM_Node::LSTM_Node(int number_maps, int map_size, bool batch_normalization) {
	number_nodes = number_maps * map_size;

	if (batch_normalization) {
		for (int h = 0; h < number_node_types - 1; h++) {
			this->batch_normalization[h][0] = new Batch_Normalization(number_maps, map_size);
			this->batch_normalization[h][1] = new Batch_Normalization(number_maps, map_size);
		}
		this->batch_normalization[cell_output][0] = new Batch_Normalization(number_maps, map_size);
		this->batch_normalization[cell_output][1] = nullptr;
	}
	else {
		for (int h = 0; h < number_node_types; h++) {
			this->batch_normalization[h][0] = nullptr;
			this->batch_normalization[h][1] = nullptr;
		}
	}
	for (int h = 0; h < number_weight_types; h++) {
		cudaMalloc(&bias[h], sizeof(float) * number_maps);
		bias_optimizer[h] = new Optimizer();
		bias_optimizer[h]->Resize_Memory(number_maps);
	}
	for (int h = 0; h < number_weight_types - 1; h++) {
		cudaMalloc(&peephole[h], sizeof(float) * number_maps);
		peephole_optimizer[h] = new Optimizer();
		peephole_optimizer[h]->Resize_Memory(number_maps);
	}
	for (int h = 0; h < number_node_types; h++) {
		cudaMalloc(&error[h][0], sizeof(float) * number_nodes);
		cudaMalloc(&error[h][1], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[h][0], sizeof(float) * number_nodes);
		cudaMalloc(&neuron[h][1], sizeof(float) * number_nodes);
	}
}
LSTM_Node::~LSTM_Node() {}

void LSTM_Node::Destroy() {
	for (int h = 0; h < number_node_types; h++) {
		if (batch_normalization[h][0]) {
			batch_normalization[h][0]->Destroy();
			delete batch_normalization[h][0];
		}
		if (batch_normalization[h][1]) {
			batch_normalization[h][1]->Destroy();
			delete batch_normalization[h][1];
		}
		cudaFree(error[h][0]);
		cudaFree(error[h][1]);
		cudaFree(neuron[h][0]);
		cudaFree(neuron[h][1]);
	}
	for (int h = 0; h < number_weight_types; h++) {
		cudaFree(bias[h]);
		bias_optimizer[h]->Destroy();
		delete bias_optimizer[h];
	}
	for (int h = 0; h < number_weight_types - 1; h++) {
		cudaFree(peephole[h]);
		peephole_optimizer[h]->Destroy();
		delete peephole_optimizer[h];
	}
}
void LSTM_Node::Resize_Memory(int batch_size, int time_step) {
	for (int h = 0; h < number_node_types; h++) {
		int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

		if (batch_normalization[h][0]) batch_normalization[h][0]->Resize_Memory(batch_size, time_step);
		if (batch_normalization[h][1]) batch_normalization[h][1]->Resize_Memory(batch_size, time_step);

		cudaFree(error[h][0]);
		cudaFree(error[h][1]);
		cudaFree(neuron[h][0]);
		cudaFree(neuron[h][1]);

		cudaMalloc(&error[h][0], memory_size);
		cudaMalloc(&error[h][1], memory_size);
		cudaMalloc(&neuron[h][0], memory_size);
		cudaMalloc(&neuron[h][1], memory_size);
	}
}


LSTM_Weight::LSTM_Weight(int number_weights) {
	for (int h = 0; h < number_weight_types; h++) {
		optimizer[h] = new Optimizer();
		optimizer[h]->Resize_Memory(number_weights);
		cudaMalloc(&weight[h], sizeof(float) * number_weights);
	}
}
LSTM_Weight::~LSTM_Weight() {}

void LSTM_Weight::Destroy() {
	for (int h = 0; h < number_weight_types; h++) {
		optimizer[h]->Destroy();
		delete optimizer[h];
		cudaFree(weight[h]);
	}
}


void Neural_Networks::Activate(Layer *layer, string phase, int time_index) {
	if (layer->batch_normalization[0]) layer->batch_normalization[0]->Activate(phase, layer->neuron[0], time_index);
	if (layer->batch_normalization[1]) layer->batch_normalization[1]->Activate(phase, layer->neuron[1], time_index);

	::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *layer);

	if (strstr(layer->properties.c_str(), "LSTM")) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);
		bool backward = (strstr(layer->properties.c_str(), "backward") != 0);

		LSTM_Node *LSTM_node = layer->LSTM_node;

		for (int i = 0; i < LSTM_node->number_node_types - 1; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Activate(phase, LSTM_node->neuron[i][0], time_index);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Activate(phase, LSTM_node->neuron[i][1], time_index);
		}

		::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (nullptr), *layer, *LSTM_node, backward);

		if (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) {
			LSTM_node->batch_normalization[LSTM_node->cell_output][0]->Activate(phase, LSTM_node->neuron[LSTM_node->cell_output][0], time_index);
		}
		::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, nullptr, *layer, *LSTM_node, backward);
	}
	else if (strstr(layer->properties.c_str(), "RNN")) {
		::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (6, time_index, *layer);
	}
	else {
		if (strstr(layer->properties.c_str(), "ELU")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "PReLU")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (2, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "ReLU")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (3, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "sigmoid")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (4, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "softmax")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (5, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "tangent")) {
			::Activate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (6, time_index, *layer);
		}

		if (strstr(layer->properties.c_str(), "dropout")) {
			double rate = atof(strstr(layer->properties.c_str(), "dropout") + 7);

			Dropout << <batch_size * layer->number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (phase == "training", time_index, rate, *layer);
		}
	}
}
void Neural_Networks::Adjust_Parameter(Layer *layer, double gradient_clip, double learning_rate) {
	for (int g = 0; g < layer->parent_layer.size(); g++) {
		if (layer->connection[g]->properties[0] == 'W') {
			Connection *connection = layer->connection[g];

			int number_blocks = (connection->number_weights / NUMBER_THREADS + 1 > 65536) ? (65536) : (connection->number_weights / NUMBER_THREADS + 1);

			if (connection->LSTM_weight) {
				for (int i = 0; i < connection->LSTM_weight->number_weight_types; i++) {
					::Adjust_Parameter << <number_blocks, NUMBER_THREADS >> > (connection->number_weights, connection->LSTM_weight->weight[i], gradient_clip, learning_rate, *connection->LSTM_weight->optimizer[i]);
				}
			}
			else {
				::Adjust_Parameter << <number_blocks, NUMBER_THREADS >> > (connection->number_weights, connection->weight, gradient_clip, learning_rate, *connection->optimizer);
			}
		}
	}

	if (layer->LSTM_node) {
		LSTM_Node *LSTM_node = layer->LSTM_node;

		for (int i = 0; i < LSTM_node->number_weight_types; i++) {
			::Adjust_Parameter << <layer->number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (layer->number_maps, LSTM_node->bias[i], gradient_clip, learning_rate, *LSTM_node->bias_optimizer[i]);
		}
		for (int i = 0; i < LSTM_node->number_weight_types - 1; i++) {
			::Adjust_Parameter << <layer->number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (layer->number_maps, LSTM_node->peephole[i], gradient_clip, learning_rate, *LSTM_node->peephole_optimizer[i]);
		}
		for (int i = 0; i < LSTM_node->number_node_types; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Adjust_Parameter(gradient_clip, learning_rate);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Adjust_Parameter(gradient_clip, learning_rate);
		}
	}
	if (layer->bias_optimizer) {
		::Adjust_Parameter << < layer->number_maps / NUMBER_THREADS + 1, NUMBER_THREADS >> > (layer->number_maps, layer->bias, gradient_clip, learning_rate, *layer->bias_optimizer);
	}
	if (layer->slope_optimizer) {
		::Adjust_Parameter << < layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (layer->number_nodes, layer->slope, gradient_clip, learning_rate, *layer->slope_optimizer);
	}
	if (layer->batch_normalization[0]) {
		layer->batch_normalization[0]->Adjust_Parameter(gradient_clip, learning_rate);
	}
	if (layer->batch_normalization[1]) {
		layer->batch_normalization[1]->Adjust_Parameter(gradient_clip, learning_rate);
	}
}
void Neural_Networks::Backpropagate(Layer *layer, int time_index, bool backward) {
	int t = time_index;

	for (int g = 0; g < layer->parent_layer.size(); g++) {
		bool recurrent = (strstr(layer->connection[g]->properties.c_str(), "recurrent") != 0);

		Connection *connection = layer->connection[g];

		Layer *parent_layer = layer->parent_layer[g];

		if (recurrent || parent_layer->Check_Mask(t)) {
			if (connection->properties[0] == 'P') {
				::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *connection, *layer, *parent_layer, backward);
			}
			else if (connection->properties[0] == 'W') {
				if (recurrent == false) {
					if (layer->LSTM_node) {
						::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *connection, *layer, *parent_layer, *layer->LSTM_node, *connection->LSTM_weight, backward);
					}
					else {
						::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *connection, *layer, *parent_layer, backward);
					}
				}
				else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
					if (layer->LSTM_node) {
						::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *connection, *layer, *parent_layer, *layer->LSTM_node, *connection->LSTM_weight, backward);
					}
					else {
						::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (2, time_index, *connection, *layer, *parent_layer, backward);
					}
				}
			}
			else if (strstr(connection->properties.c_str(), "add")) {
				::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (3, time_index, *connection, *layer, *parent_layer, backward);
			}
			else if (strstr(connection->properties.c_str(), "dilate")) {
				::Backpropagate << <batch_size * parent_layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (4, time_index, *connection, *layer, *parent_layer, backward);
			}
		}
	}
}
void Neural_Networks::Feedforward(Layer *layer, int time_index, bool backward) {
	int t = time_index;

	for (int g = 0; g < layer->parent_layer.size(); g++) {
		bool recurrent = (strstr(layer->connection[g]->properties.c_str(), "recurrent") != 0);

		Connection *connection = layer->connection[g];

		Layer *parent_layer = layer->parent_layer[g];

		if (recurrent || parent_layer->Check_Mask(t)) {
			if (connection->properties[0] == 'P') {
				if (strstr(connection->properties.c_str(), "average")) {
					::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *connection, *layer, *parent_layer, backward);
				}
				else if (strstr(connection->properties.c_str(), "max")) {
					::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *connection, *layer, *parent_layer, backward);
				}
			}
			else if (connection->properties[0] == 'W') {
				if (recurrent == false) {
					if (layer->LSTM_node) {
						::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *connection, *layer, *parent_layer, *layer->LSTM_node, *connection->LSTM_weight, backward);
					}
					else {
						::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (2, time_index, *connection, *layer, *parent_layer, backward);
					}
				}
				else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
					if (layer->LSTM_node) {
						::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *connection, *layer, *parent_layer, *layer->LSTM_node, *connection->LSTM_weight, backward);
					}
					else {
						::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (3, time_index, *connection, *layer, *parent_layer, backward);
					}
				}
			}
			else if (strstr(connection->properties.c_str(), "add")) {
				::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (4, time_index, *connection, *layer, *parent_layer, backward);
			}
			else if (strstr(connection->properties.c_str(), "dilate")) {
				::Feedforward << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (5, time_index, *connection, *layer, *parent_layer, backward);
			}
		}
	}
}

void Neural_Networks::FloatToNode(float **memory, vector<Layer*> &layer) {
	for (int i = 0; i < layer.size(); i++) {
		cudaMemcpy(layer[i]->neuron[0], memory[i], sizeof(float) * batch_size * time_step * layer[i]->number_nodes, cudaMemcpyDeviceToDevice);
	}
}
void Neural_Networks::FloatToNode(float ***memory, vector<Layer*> &layer, int length_data[]) {
	for (int i = 0; i < layer.size(); i++) {
		for (int h = 0; h < batch_size; h++) {
			cudaMemset(&layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], 0, sizeof(float) * time_step * layer[i]->number_nodes);
			cudaMemcpy(&layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], memory[h][i], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[h])) *layer[i]->number_nodes, cudaMemcpyHostToDevice);
		}
	}
}
void Neural_Networks::NodeToFloat(vector<Layer*> &layer, float ***memory) {
	for (int i = 0; i < layer.size(); i++) {
		for (int h = 0; h < batch_size; h++) {
			cudaMemcpy(memory[h][i], &layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], sizeof(float) * time_step * layer[i]->number_nodes, cudaMemcpyDeviceToHost);
		}
	}
}
void Neural_Networks::Resize_Memory(int batch_size, int time_step) {
	if (time_step == 0) time_step = this->time_step;

	if (this->batch_size != batch_size || this->time_step != time_step) {
		for (int i = 0; i < layer_height; i++) {
			for (int j = 0; j < layer[i].size(); j++) {
				layer[i][j]->Resize_Memory(batch_size, time_step);
			}
		}
		this->batch_size = batch_size;
		this->time_step = time_step;
	}
}
void Neural_Networks::Zero_Memory() {
	for (int i = 1; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			int memory_size = sizeof(float) * batch_size * time_step * layer[i][j]->number_nodes;

			Layer *layer = this->layer[i][j];

			if (layer->neuron[0]) {
				cudaMemset(layer->neuron[0], 0, memory_size);
			}
			if (layer->neuron[1]) {
				cudaMemset(layer->neuron[1], 0, memory_size);
			}
			if (layer->LSTM_node) {
				LSTM_Node *LSTM_node = layer->LSTM_node;

				for (int i = 0; i < LSTM_node->number_weight_types; i++) {
					cudaMemset(LSTM_node->neuron[i][0], 0, memory_size);
					cudaMemset(LSTM_node->neuron[i][1], 0, memory_size);
				}
			}
			cudaMemset(layer->error[0], 0, memory_size);
		}
	}
}

double Neural_Networks::Calculate_Gradient(Layer *layer, double learning_rate, bool backward) {
	float sum_gradient[2] = { 0, }, *gradient;

	for (int g = 0; g < layer->parent_layer.size(); g++) {
		if (layer->connection[g]->properties[0] == 'W') {
			bool recurrent = (strstr(layer->connection[g]->properties.c_str(), "recurrent") != 0);

			Connection *connection = layer->connection[g];

			Layer *parent_layer = layer->parent_layer[g];

			if (connection->LSTM_weight) {
				int number_weights = connection->LSTM_weight->number_weight_types * connection->number_weights;

				LSTM_Optimizer optimizer = { *connection->LSTM_weight->optimizer[0], *connection->LSTM_weight->optimizer[1], *connection->LSTM_weight->optimizer[2], *connection->LSTM_weight->optimizer[3] };

				cudaMalloc(&gradient, sizeof(float) * number_weights);

				::Calculate_Gradient << <(number_weights > 65536) ? (65536) : (number_weights), NUMBER_THREADS >> > (gradient, learning_rate, *connection, *layer, *parent_layer, *layer->LSTM_node, *connection->LSTM_weight, optimizer, recurrent, backward);
				Merge << <1, NUMBER_THREADS >> > (number_weights, gradient);
				cudaMemcpy(&sum_gradient[1], gradient, sizeof(float), cudaMemcpyDeviceToHost);
				sum_gradient[0] += sum_gradient[1];

				cudaFree(gradient);
			}
			else {
				int number_weights = connection->number_weights;

				cudaMalloc(&gradient, sizeof(float) * number_weights);

				::Calculate_Gradient << <(number_weights > 65536) ? (65536) : (number_weights), NUMBER_THREADS >> > (gradient, learning_rate, *connection, *layer, *parent_layer, *connection->optimizer, recurrent, backward);
				Merge << <1, NUMBER_THREADS >> > (number_weights, gradient);
				cudaMemcpy(&sum_gradient[1], gradient, sizeof(float), cudaMemcpyDeviceToHost);
				sum_gradient[0] += sum_gradient[1];

				cudaFree(gradient);
			}
		}
	}

	if (layer->LSTM_node) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);

		int number_parameters = layer->LSTM_node->number_weight_types * layer->number_maps;

		LSTM_Node *LSTM_node = layer->LSTM_node;

		LSTM_Optimizer bias_optimizer = { *LSTM_node->bias_optimizer[0], *LSTM_node->bias_optimizer[1], *LSTM_node->bias_optimizer[2], *LSTM_node->bias_optimizer[3] };
		LSTM_Optimizer peephole_optimizer = { *LSTM_node->peephole_optimizer[0], *LSTM_node->peephole_optimizer[1] , *LSTM_node->peephole_optimizer[2] };

		cudaMalloc(&gradient, sizeof(float) * number_parameters);

		::Calculate_Gradient << <number_parameters, NUMBER_THREADS >> > (gradient, (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->error_backup) : (nullptr), (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (nullptr), learning_rate, *layer, *LSTM_node, bias_optimizer, peephole_optimizer, backward);
		Merge << <1, NUMBER_THREADS >> > (number_parameters, gradient);
		cudaMemcpy(&sum_gradient[1], gradient, sizeof(float), cudaMemcpyDeviceToHost);
		sum_gradient[0] += sum_gradient[1];

		cudaFree(gradient);

		for (int i = 0; i < LSTM_node->number_node_types; i++) {
			if (LSTM_node->batch_normalization[i][0]) sum_gradient[0] += LSTM_node->batch_normalization[i][0]->Calculate_Gradient(learning_rate);
			if (LSTM_node->batch_normalization[i][1]) sum_gradient[0] += LSTM_node->batch_normalization[i][1]->Calculate_Gradient(learning_rate);
		}
	}
	if (layer->bias_optimizer) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);

		int number_parameters = layer->number_maps;

		cudaMalloc(&gradient, sizeof(float) * number_parameters);

		::Calculate_Gradient << <number_parameters, NUMBER_THREADS >> > (gradient, (batch_normalization) ? (layer->batch_normalization[0]->error_backup) : (nullptr), learning_rate, *layer, *layer->bias_optimizer);
		Merge << <1, NUMBER_THREADS >> > (number_parameters, gradient);
		cudaMemcpy(&sum_gradient[1], gradient, sizeof(float), cudaMemcpyDeviceToHost);
		sum_gradient[0] += sum_gradient[1];

		cudaFree(gradient);
	}
	if (layer->slope_optimizer) {
		int number_parameters = layer->number_nodes;

		cudaMalloc(&gradient, sizeof(float) * number_parameters);

		::Calculate_Gradient << <number_parameters, NUMBER_THREADS >> > (gradient, learning_rate, *layer, *layer->slope_optimizer);
		Merge << <1, NUMBER_THREADS >> > (number_parameters, gradient);
		cudaMemcpy(&sum_gradient[1], gradient, sizeof(float), cudaMemcpyDeviceToHost);
		sum_gradient[0] += sum_gradient[1];

		cudaFree(gradient);
	}
	if (layer->batch_normalization[0]) {
		sum_gradient[0] += layer->batch_normalization[0]->Calculate_Gradient(learning_rate);
	}
	if (layer->batch_normalization[1]) {
		sum_gradient[0] += layer->batch_normalization[1]->Calculate_Gradient(learning_rate);
	}
	return sum_gradient[0];
}
double Neural_Networks::Differentiate(Layer *layer, float target_output[], int time_index) {
	int t = time_index;

	float sum = 0;

	if (target_output && layer->Check_Mask(t) && (strstr(layer->properties.c_str(), "CE") || strstr(layer->properties.c_str(), "MSE"))) {
		float *loss;

		cudaMalloc(&loss, sizeof(float) * batch_size * layer->number_nodes);

		::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > ((strstr(layer->properties.c_str(), "CE")) ? (0) : (1), time_index, loss, target_output, *layer);
		Merge << <1, NUMBER_THREADS >> > (batch_size * layer->number_nodes, loss);
		cudaMemcpy(&sum, loss, sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(loss);
	}

	if (strstr(layer->properties.c_str(), "LSTM")) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);
		bool backward = (strstr(layer->properties.c_str(), "backward") != 0);

		LSTM_Node *LSTM_node = layer->LSTM_node;

		float *input_error_backup = (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->input][0]->error_backup) : (nullptr);
		float *forget_error_backup = (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->forget][0]->error_backup) : (nullptr);
		float *output_error_backup = (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->output][0]->error_backup) : (nullptr);
		float *cell_output_neuron_backup = (batch_normalization) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (nullptr);

		::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *layer, *LSTM_node, backward);

		if (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) {
			LSTM_node->batch_normalization[LSTM_node->cell_output][0]->Differentiate(LSTM_node->error[LSTM_node->cell_output][0], time_index);
		}
		::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *layer, *LSTM_node, backward, input_error_backup, forget_error_backup, output_error_backup, cell_output_neuron_backup);

		for (int i = 0; i < LSTM_node->number_node_types - 1; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Differentiate(LSTM_node->error[i][0], time_index);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Differentiate(LSTM_node->error[i][1], time_index);
		}
	}
	else if (strstr(layer->properties.c_str(), "RNN")) {
		::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (4, time_index, *layer);
	}
	else {
		if (strstr(layer->properties.c_str(), "ELU")) {
			::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (0, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "PReLU")) {
			::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (1, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "ReLU")) {
			::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (2, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "sigmoid") && !strstr(layer->properties.c_str(), "CE")) {
			::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (3, time_index, *layer);
		}
		else if (strstr(layer->properties.c_str(), "softmax")) {
			// error = error;
		}
		else if (strstr(layer->properties.c_str(), "tangent")) {
			::Differentiate << <batch_size * layer->number_nodes / NUMBER_THREADS + 1, NUMBER_THREADS >> > (4, time_index, *layer);
		}
	}

	if (layer->error[1]) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * layer->number_nodes;

			cudaMemcpy(&layer->error[1][index], &layer->error[0][index], sizeof(float) * layer->number_nodes, cudaMemcpyDeviceToDevice);
		}
	}
	if (layer->batch_normalization[0]) {
		layer->batch_normalization[0]->Differentiate(layer->error[0], time_index);
	}
	if (layer->batch_normalization[1]) {
		layer->batch_normalization[1]->Differentiate(layer->error[1], time_index);
	}
	return sum;
}
double Neural_Networks::Differentiate(Layer *layer, int length_data[], vector<string> target_label_sequence[]) {
	double sum = 0;

	if (CTC && strstr(layer->properties.c_str(), "CTC")) {
		double *log_likelihood = new double[batch_size];

		vector<string> *label_sequence = new vector<string>[batch_size];

		for (int h = 0; h < batch_size; h++) {
			for (int j = 0; j < target_label_sequence[h].size(); j++) {
				label_sequence[h].push_back("");
				label_sequence[h].push_back(target_label_sequence[h][j]);
			}
			label_sequence[h].push_back("");
		}
		CTC->Calculate_Error(label_sequence, batch_size, time_step, length_data, layer->error[0], layer->neuron[0], log_likelihood);

		for (int h = 0; h < batch_size; h++) {
			sum += log_likelihood[h];
		}
		delete[] label_sequence;
		delete[] log_likelihood;
	}
	for (int t = time_step - 1; t >= 0; t--) {
		// Differentiate(layer, nullptr, t);
	}
	return sum;
}

Neural_Networks::Neural_Networks(string path) {
	ifstream file(path);

	if (file.is_open()) {
		int number_connections;
		int number_layers;

		vector<Connection*> connection;
		vector<Layer*> layer;

		file >> epsilon;
		file >> number_connections;
		file >> number_layers;
		file >> time_step;
		gradient_threshold = 0;
		CTC = nullptr;

		for (int i = 0, index, map_depth, map_height, map_width, mask, number_maps; i < number_layers; i++) {
			string properties;

			file >> map_depth;
			file >> map_height;
			file >> map_width;
			file >> number_maps;
			getline(file, properties);
			getline(file, properties);
			file >> index;

			layer.push_back(Add(new Layer(properties, number_maps, map_width, map_height, map_depth), index));

			file >> mask;

			if (mask) {
				bool *time_mask = new bool[time_step];

				for (int t = 0; t < time_step; t++) {
					file >> time_mask[t];
				}
				layer.back()->Set_Time_Mask(time_mask);
			}
		}
		for (int i = 0, index[4]; i < number_connections; i++) {
			string properties;

			file >> index[0];
			file >> index[1];
			file >> index[2];
			file >> index[3];
			getline(file, properties);
			getline(file, properties);

			connection.push_back(this->layer[index[0]][index[1]]->Connect(this->layer[index[2]][index[3]], properties));
		}
		layer_height = static_cast<int>(this->layer.size());
		Resize_Memory(1, time_step);
		Set_Epsilon(epsilon);

		for (int i = 0; i < number_layers; i++) {
			layer[i]->Load(file);
		}
		for (int i = 0; i < number_connections; i++) {
			connection[i]->Load(file);
		}
		file.close();
	}
	else {
		cerr << "[Neural_Networks], " + path + " not found" << endl;
	}
}
Neural_Networks::Neural_Networks(int time_step) {
	this->time_step = time_step;

	batch_size = 0;
	gradient_threshold = 0;
	layer_height = 0;
	CTC = nullptr;
}
Neural_Networks::~Neural_Networks() {
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			layer[i][j]->Destroy();
			delete layer[i][j];
		}
	}
	if (CTC) {
		delete CTC;
	}
}

void Neural_Networks::Decode(int length_event, float likelihood[], vector<string> &label_sequence, bool space_between_labels) {
	Decode(length_event, likelihood, label_sequence, 0, space_between_labels);
}
void Neural_Networks::Decode(int length_event, float likelihood[], vector<string> &label_sequence, int k, bool space_between_labels) {
	if (k == 0) {
		CTC->Best_Path_Decoding(length_event, likelihood, label_sequence, space_between_labels);
	}
	else {
		CTC->Prefix_Beam_Search_Decoding(length_event, likelihood, label_sequence, k, space_between_labels);
	}
}
void Neural_Networks::Initialize(double scale, double gamma) {
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			layer[i][j]->Initialize(scale, gamma);
		}
	}
}
void Neural_Networks::Save(string path) {
	int number_connections = 0;
	int number_layers = 0;

	ofstream file(path);

	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			number_connections += layer[i][j]->number_connections;
		}
		number_layers += static_cast<int>(layer[i].size());
	}
	file << epsilon << endl;
	file << number_connections << endl;
	file << number_layers << endl;
	file << time_step << endl;

	// layer definition
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			Layer *layer = this->layer[i][j];

			file << layer->map_depth << endl;
			file << layer->map_height << endl;
			file << layer->map_width << endl;
			file << layer->number_maps << endl;
			file << layer->properties << endl;
			file << i << endl;

			if (layer->time_mask) {
				file << 1 << endl;

				for (int t = 0; t < time_step; t++) {
					file << layer->time_mask[t] << endl;
				}
			}
			else {
				file << 0 << endl;
			}
			file << endl;
		}
	}

	// connection definition
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			Layer *layer = this->layer[i][j];

			for (int k = 0; k < layer->number_connections; k++) {
				file << i << " " << j << " " << layer->parent_layer[k]->index[0] << " " << layer->parent_layer[k]->index[1] << endl;
				file << layer->connection[k]->properties << endl << endl;
			}
		}
	}

	// layer parameter
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			layer[i][j]->Save(file);
		}
	}

	// connection parameter
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			for (int k = 0; k < layer[i][j]->number_connections; k++) {
				layer[i][j]->connection[k]->Save(file);
			}
		}
	}
	file.close();
}
void Neural_Networks::Set_CTC_Loss(int number_labels, string label[]) {
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			if (strstr(layer[i][j]->properties.c_str(), "CTC")) {
				if (number_labels != layer[i][j]->number_nodes) {
					cerr << "[Set_CTC_Loss], number_labels != number_nodes" << endl;
					return;
				}
				if (CTC) {
					delete CTC;
				}
				CTC = new Connectionist_Temporal_Classification(number_labels, label);
				return;
			}
		}
	}
	cerr << "[Set_CTC_Loss], there is no layer with CTC loss" << endl;
}
void Neural_Networks::Set_Epsilon(double epsilon) {
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			layer[i][j]->Set_Epsilon(epsilon);
		}
	}
	this->epsilon = epsilon;
}
void Neural_Networks::Set_Gradient_Threshold(double gradient_threshold) {
	this->gradient_threshold = gradient_threshold;
}
void Neural_Networks::Set_Optimizer(Optimizer *optimizer) {
	if (optimizer == nullptr) {
		cerr << "[Set_Optimizer], optimizer = nullptr" << endl;
		return;
	}

	for (int i = 1; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			layer[i][j]->Set_Optimizer(optimizer->Copy());
		}
	}
	delete optimizer;
}
void Neural_Networks::Test(float input[], float output[], int _length_data) {
	int *length_data = (_length_data == 0) ? (nullptr) : (&_length_data);

	Test(1, &input, &output, length_data);
}
void Neural_Networks::Test(int batch_size, float **_input, float **_output, int length_data[]) {
	float ***input = new float**[batch_size];
	float ***output = new float**[batch_size];

	for (int h = 0; h < batch_size; h++) {
		input[h] = &_input[h];
		output[h] = &_output[h];
	}
	Test(batch_size, input, output, length_data);

	delete[] input;
	delete[] output;
}
void Neural_Networks::Test(int batch_size, float ***input, float ***output, int length_data[]) {
	Resize_Memory(batch_size);
	FloatToNode(input, layer[0], length_data);
	Zero_Memory();

	for (int i = 1; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			Layer *layer = this->layer[i][j];

			if (strstr(layer->properties.c_str(), "backward")) {
				for (int t = time_step - 1; t >= 0; t--) {
					Feedforward(layer, t, true);
					Activate(layer, "inference", t);
				}
			}
			else {
				for (int t = 0; t < time_step; t++) {
					Feedforward(layer, t);
					Activate(layer, "inference", t);
				}
			}
		}
	}
	NodeToFloat(layer[layer_height - 1], output);
}

double Neural_Networks::Train(int batch_size, int number_training, float **input, float **target_output, double epsilon, double learning_rate, double noise_standard_deviation) {
	return Train(batch_size, number_training, nullptr, input, target_output, learning_rate, epsilon, noise_standard_deviation);
}
double Neural_Networks::Train(int batch_size, int number_training, int length_data[], float **_input, float **_target_output, double epsilon, double learning_rate, double noise_standard_deviation) {
	double loss;

	float ***input = new float**[number_training];
	float ***target_output = new float**[number_training];

	for (int h = 0; h < number_training; h++) {
		input[h] = &_input[h];
		target_output[h] = &_target_output[h];
	}
	loss = Train(batch_size, number_training, length_data, input, target_output, nullptr, learning_rate, epsilon, noise_standard_deviation);

	delete[] input;
	delete[] target_output;

	return loss;
}
double Neural_Networks::Train(int batch_size, int number_training, float **input, vector<string> target_label_sequence[], double learning_rate, double epsilon, double noise_standard_deviation) {
	return Train(batch_size, number_training, nullptr, input, target_label_sequence, learning_rate, epsilon, noise_standard_deviation);
}
double Neural_Networks::Train(int batch_size, int number_training, int length_data[], float **_input, vector<string> target_label_sequence[], double learning_rate, double epsilon, double noise_standard_deviation) {
	double loss;

	float ***input = new float**[number_training];

	for (int h = 0; h < number_training; h++) {
		input[h] = &_input[h];
	}
	loss = Train(batch_size, number_training, length_data, input, nullptr, target_label_sequence, learning_rate, epsilon, noise_standard_deviation);

	delete[] input;

	return loss;
}
double Neural_Networks::Train(int batch_size, int number_training, int length_data[], float ***input, float ***target_output, vector<string> target_label_sequence[], double learning_rate, double epsilon, double noise_standard_deviation) {
	int *index = new int[number_training];
	int *length_data_batch = (target_label_sequence && length_data) ? (new int[batch_size]) : (nullptr);

	float **input_batch = new float*[layer[0].size()];
	float **target_output_batch = (target_output) ? (new float*[layer[layer_height - 1].size()]) : (nullptr);

	double sum = 0;

	vector<string> *target_label_sequence_batch = (target_label_sequence) ? (new vector<string>[batch_size]) : (nullptr);

	for (int i = 0; i < number_training; i++) {
		index[i] = i;
	}
	for (int i = 0; i < number_training; i++) {
		int j = rand() % number_training;
		int k = index[i];

		index[i] = index[j];
		index[j] = k;
	}

	for (int i = 0, j = 0; j < layer[i].size(); j++) {
		cudaMalloc(&input_batch[j], sizeof(float) * batch_size * time_step * layer[i][j]->number_nodes);
	}
	for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
		cudaMalloc(&target_output_batch[j], sizeof(float) * batch_size * time_step * layer[i][j]->number_nodes);
	}
	Resize_Memory(batch_size);
	Set_Epsilon(epsilon);

	for (int g = 0, h = 0; g < number_training; g++) {
		for (int i = 0, j = 0; j < layer[i].size(); j++) {
			cudaMemset(&input_batch[j][h * time_step * layer[i][j]->number_nodes], 0, sizeof(float) * time_step * layer[i][j]->number_nodes);
			cudaMemcpy(&input_batch[j][h * time_step * layer[i][j]->number_nodes], input[index[g]][j], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[index[g]])) * layer[i][j]->number_nodes, cudaMemcpyHostToDevice);
		}
		for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
			cudaMemset(&target_output_batch[j][h * time_step * layer[i][j]->number_nodes], 0, sizeof(float) * time_step * layer[i][j]->number_nodes);
			cudaMemcpy(&target_output_batch[j][h * time_step * layer[i][j]->number_nodes], target_output[index[g]][j], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[index[g]])) * layer[i][j]->number_nodes, cudaMemcpyHostToDevice);
		}
		if (target_label_sequence) {
			if (length_data) {
				length_data_batch[h] = length_data[index[g]];
			}
			target_label_sequence_batch[h] = target_label_sequence[index[g]];
		}

		if (++h == batch_size) {
			double sum_gradient = 0, gradient_clip = 1;

			if (noise_standard_deviation) {
				for (int i = 0, j = 0; j < layer[i].size(); j++) {
					int memory_size = batch_size * time_step * layer[i][j]->number_nodes;
					int number_blocks = (memory_size / NUMBER_THREADS + 1 > 65536) ? (65536) : (memory_size / NUMBER_THREADS + 1);

					float *noise;

					cudaMalloc(&noise, sizeof(float) * memory_size);

					Random_Normal << <number_blocks, NUMBER_THREADS >> > (memory_size, noise, noise_standard_deviation, rand());
					::Add << <number_blocks, NUMBER_THREADS >> > (memory_size, input_batch[j], noise, input_batch[j]);
					cudaFree(noise);
				}
			}

			for (int i = 1; i < layer_height; i++) {
				for (int j = 0; j < layer[i].size(); j++) {
					Layer *layer = this->layer[i][j];

					// initialize dropout mask
					if (strstr(layer->properties.c_str(), "dropout")) {
						bool *mask = new bool[batch_size * layer->number_maps];

						double rate = atof(strstr(layer->properties.c_str(), "dropout") + 7);

						for (int k = 0; k < batch_size * layer->number_maps; k++) {
							mask[k] = ((double)rand() / RAND_MAX <= rate);
						}
						cudaMemcpy(layer->dropout_mask, mask, batch_size * layer->number_maps, cudaMemcpyHostToDevice);
						delete[] mask;
					}
				}
			}
			FloatToNode(input_batch, layer[0]);
			Zero_Memory();

			// forward propagation
			for (int i = 1; i < layer_height; i++) {
				for (int j = 0; j < layer[i].size(); j++) {
					Layer *layer = this->layer[i][j];

					if (strstr(layer->properties.c_str(), "backward")) {
						for (int t = time_step - 1; t >= 0; t--) {
							Feedforward(layer, t, true);
							Activate(layer, "training", t);
						}
					}
					else {
						for (int t = 0; t < time_step; t++) {
							Feedforward(layer, t);
							Activate(layer, "training", t);
						}
					}
				}
			}

			// error backpropagation
			for (int i = layer_height - 1; i > 0; i--) {
				for (int j = 0; j < layer[i].size(); j++) {
					Layer *layer = this->layer[i][j];

					if (strstr(layer->properties.c_str(), "CTC")) {
						sum += Differentiate(layer, length_data_batch, target_label_sequence_batch);

						for (int t = 0; t < time_step; t++) {
							Backpropagate(layer, t);
						}
					}
					else {
						if (strstr(layer->properties.c_str(), "backward")) {
							for (int t = 0; t < time_step; t++) {
								sum += Differentiate(layer, (i == layer_height - 1) ? (target_output_batch[j]) : (nullptr), t);
								Backpropagate(layer, t, true);
							}
						}
						else {
							for (int t = time_step - 1; t >= 0; t--) {
								sum += Differentiate(layer, (i == layer_height - 1) ? (target_output_batch[j]) : (nullptr), t);
								Backpropagate(layer, t);
							}
						}
					}
				}
			}

			// calculate gradient
			for (int i = layer_height - 1; i > 0; i--) {
				for (int j = 0; j < layer[i].size(); j++) {
					sum_gradient += Calculate_Gradient(layer[i][j], learning_rate, strstr(layer[i][j]->properties.c_str(), "backward") != 0);
				}
			}
			if (gradient_threshold && sqrt(sum_gradient) > gradient_threshold) {
				gradient_clip = gradient_threshold / sqrt(sum_gradient);
			}

			// adjust parameter
			for (int i = layer_height - 1; i > 0; i--) {
				for (int j = 0; j < layer[i].size(); j++) {
					Adjust_Parameter(layer[i][j], gradient_clip, learning_rate);
				}
			}
			h = 0;
		}
	}

	// calculate batch mean and variance
	for (int i = 0; i < layer_height; i++) {
		for (int j = 0; j < layer[i].size(); j++) {
			if (layer[i][j]->LSTM_node) {
				LSTM_Node *LSTM_node = layer[i][j]->LSTM_node;

				for (int h = 0; h < LSTM_node->number_node_types; h++) {
					if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Calculate_Mean_Variance(number_training / batch_size);
					if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Calculate_Mean_Variance(number_training / batch_size);
				}
			}
			if (layer[i][j]->batch_normalization[0]) layer[i][j]->batch_normalization[0]->Calculate_Mean_Variance(number_training / batch_size);
			if (layer[i][j]->batch_normalization[1]) layer[i][j]->batch_normalization[1]->Calculate_Mean_Variance(number_training / batch_size);
		}
	}

	for (int i = 0, j = 0; j < layer[i].size(); j++) {
		cudaFree(input_batch[j]);
	}
	if (target_output) {
		for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
			cudaFree(target_output_batch[j]);
		}
		delete[] target_output_batch;
	}
	if (target_label_sequence) {
		if (length_data) {
			delete[] length_data_batch;
		}
		delete[] target_label_sequence_batch;
	}
	delete[] index;
	delete[] input_batch;

	return sum / number_training;
}

Layer* Neural_Networks::Add(Layer *layer, int index) {
	if (index < 0) {
		index = static_cast<int>(this->layer.size());
	}
	while (this->layer.size() <= index) {
		vector<Layer*> layer_holder;

		this->layer.push_back(layer_holder);
		layer_height++;
	}
	layer->index[0] = index;
	layer->index[1] = static_cast<int>(this->layer[index].size());
	this->layer[index].push_back(layer);

	return layer;
}
Layer* Neural_Networks::Get_Layer(int y, int x) {
	if (y >= layer_height || static_cast<int>(layer[y].size()) <= x) {
		return nullptr;
	}
	return layer[y][x];
}


void Optimizer::Initialize(string type, double epsilon, double factor_1, double factor_2) {
	string name[] = { "", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam" };

	this->epsilon = epsilon;
	this->factor_1 = factor_1;
	this->factor_2 = factor_2;
	this->type = 0;

	this->gradient = nullptr;
	this->momentum = nullptr;
	this->velocity = nullptr;

	for (int i = 0; i < 7; i++) {
		if (type == name[i]) {
			this->type = i;
			return;
		}
	}
	cerr << "[Initialize], unexpected type: " << type << endl;
}

Optimizer::Optimizer() {
	Initialize("", 0, 0, 0);
}
Optimizer::Optimizer(string type, double momentum_epsilon) {
	Initialize(type, momentum_epsilon, momentum_epsilon, 0);
}
Optimizer::Optimizer(string type, double decay_rate, double epsilon) {
	Initialize(type, epsilon, decay_rate, 0);
}
Optimizer::Optimizer(string type, double epsilon, double beta_1, double beta_2) {
	Initialize(type, epsilon, beta_1, beta_2);
}
Optimizer::~Optimizer() {}

void Optimizer::Destroy() {
	if (gradient) cudaFree(gradient);
	if (momentum) cudaFree(momentum);
	if (velocity) cudaFree(velocity);
}
void Optimizer::Resize_Memory(int number_parameters) {
	if (number_parameters == 0) {
		return;
	}
	if (gradient) {
		cudaFree(gradient);
	}
	cudaMalloc(&gradient, sizeof(float) * number_parameters);

	switch (type) {
	case 1:
	case 2:
	case 3:
	case 4:
		if (velocity) {
			cudaFree(velocity);
		}
		cudaMalloc(&velocity, sizeof(float) * number_parameters);
		cudaMemset(velocity, 0, sizeof(float) * number_parameters);
		break;
	case 5:
	case 6:
		if (momentum) {
			cudaFree(momentum);
		}
		cudaMalloc(&momentum, sizeof(float) * number_parameters);
		cudaMemset(momentum, 0, sizeof(float) * number_parameters);

		if (velocity) {
			cudaFree(velocity);
		}
		cudaMalloc(&velocity, sizeof(float) * number_parameters);
		cudaMemset(velocity, 0, sizeof(float) * number_parameters);
		break;
	}
}

Optimizer* Optimizer::Copy(int number_parameters) {
	string name[] = { "", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam" };

	Optimizer *optimizer = new Optimizer(name[type], epsilon, factor_1, factor_2);

	optimizer->Resize_Memory(number_parameters);

	return optimizer;
}