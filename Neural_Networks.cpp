#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>

#include "Neural_Networks.h"

Batch_Normalization::Batch_Normalization(){}
Batch_Normalization::Batch_Normalization(int batch_size, int map_size, int time_step) {
	Construct(batch_size, map_size, time_step);
}
Batch_Normalization::~Batch_Normalization() {
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;

	delete[] derivative_backup;
	delete[] derivative_normalized;
	delete[] neuron_backup;
	delete[] neuron_normalized;
}

void Batch_Normalization::Activate(string option, int number_nodes, int time_index, double _neuron[]) {
	int t = time_index;

	double &mean = this->mean[t];
	double &variance = this->variance[t];
	double &sum_mean = this->sum_mean[t];
	double &sum_variance = this->sum_variance[t];

	if (option == "training") {
		double sum = 0;

		for (int h = 0; h < batch_size; h++) {
			double *neuron = &_neuron[(h * time_step + t) * number_nodes];

			for (int j = 0; j < map_size; j++) {
				sum += neuron[j];
			}
		}
		sum_mean += (mean = sum / (batch_size * map_size));

		sum = 0;
		for (int h = 0; h < batch_size; h++) {
			double *neuron = &_neuron[(h * time_step + t) * number_nodes];

			for (int j = 0;j < map_size; j++) {
				sum += (neuron[j] - mean) * (neuron[j] - mean);
			}
		}
		sum_variance += (variance = sum / (batch_size * map_size));

		for (int h = 0; h < batch_size; h++) {
			int index = h * time_step + t;

			double *neuron = &_neuron[index * number_nodes];
			double *neuron_backup = &this->neuron_backup[index * map_size];
			double *neuron_normalized = &this->neuron_normalized[index * map_size];

			for (int j = 0; j < map_size; j++) {
				neuron_backup[j] = neuron[j];
				neuron_normalized[j] = (neuron[j] - mean) / sqrt(variance + epsilon);
				neuron[j] = gamma * neuron_normalized[j] + beta;
			}
		}
	}
	else if (option == "inference") {
		double stdv = sqrt(variance + epsilon);

		for (int h = 0; h < batch_size; h++) {
			int index = h * time_step + t;

			double *neuron = &_neuron[index * number_nodes];
			double *neuron_backup = &this->neuron_backup[index * map_size];

			for (int j = 0; j < map_size; j++) {
				neuron_backup[j] = neuron[j];
				neuron[j] = gamma / stdv * neuron[j] + (beta - gamma * mean / stdv);
			}
		}
	}
}
void Batch_Normalization::Adjust_Parameter() {
	double sum = 0;

	for (int h = 0; h < batch_size * time_step; h++) {
		double *derivative_backup = &this->derivative_backup[h * map_size];
		double *neuron_normalized = &this->neuron_normalized[h * map_size];

		for (int j = 0; j < map_size; j++) {
			sum += derivative_backup[j] * neuron_normalized[j];
		}
	}
	gamma -= sum;

	sum = 0;
	for (int h = 0; h < batch_size * time_step; h++) {
		double *derivative_backup = &this->derivative_backup[h * map_size];

		for (int j = 0; j < map_size; j++) {
			sum += derivative_backup[j];
		}
	}
	beta -= sum;
}
void Batch_Normalization::Calculate_Mean_Variance(int number_batches) {
	if (number_batches) {
		for (int t = 0; t < time_step; t++) {
			mean[t] = sum_mean[t] / number_batches;
			variance[t] = ((double)batch_size / (batch_size - 1)) * sum_variance[t] / number_batches;
		}
	}
	else {
		memset(sum_mean, 0, sizeof(double) * time_step);
		memset(sum_variance, 0, sizeof(double) * time_step);
	}
}
void Batch_Normalization::Construct(int batch_size, int map_size, int time_step) {
	this->batch_size = batch_size;
	this->map_size = map_size;
	this->time_step = time_step;

	mean = new double[time_step];
	variance = new double[time_step];
	sum_mean = new double[time_step];
	sum_variance = new double[time_step];

	derivative_backup = new double[batch_size * time_step * map_size];
	derivative_normalized = new double[batch_size * time_step * map_size];
	neuron_backup = new double[batch_size * time_step * map_size];
	neuron_normalized = new double[batch_size * time_step * map_size];
}
void Batch_Normalization::Differentiate(int number_nodes, int time_index, double _derivative[]) {
	int t = time_index;

	double derivative_mean;
	double derivative_variance;
	double sum = 0;

	double &mean = this->mean[t];
	double &variance = this->variance[t];

	for (int h = 0; h < batch_size; h++) {
		int index = (h * time_step + t);

		double *derivative = &_derivative[index * number_nodes];
		double *derivative_normalized = &this->derivative_normalized[index * map_size];
		double *neuron_backup = &this->neuron_backup[index * map_size];

		for (int j = 0; j < map_size; j++) {
			derivative_normalized[j] = derivative[j] * gamma;
			sum += derivative_normalized[j] * (neuron_backup[j] - mean);
		}
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, (double)-1.5);

	sum = 0;
	for (int h = 0; h < batch_size; h++) {
		double *derivative_normalized = &this->derivative_normalized[(h * time_step + t) * map_size];

		for (int j = 0; j < map_size; j++) {
			sum += derivative_normalized[j];
		}
	}
	derivative_mean = -sum / sqrt(variance + epsilon);

	for (int h = 0; h < batch_size; h++) {
		int index = (h * time_step + t);

		double *derivative = &_derivative[index * number_nodes];
		double *derivative_backup = &this->derivative_backup[index * map_size];
		double *derivative_normalized = &this->derivative_normalized[index * map_size];
		double *neuron_backup = &this->neuron_backup[index * map_size];

		for (int j = 0; j < map_size; j++) {
			derivative_backup[j] = derivative[j];
			derivative[j] = derivative_normalized[j] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_backup[j] - mean) / (batch_size * map_size) + derivative_mean / (batch_size * map_size);
		}
	}
}
void Batch_Normalization::Resize_Node(int batch_size, double epsilon) {
	if (this->batch_size != batch_size) {
		derivative_backup = (double*)realloc(derivative_backup, sizeof(double) * batch_size * time_step * map_size);
		derivative_normalized = (double*)realloc(derivative_normalized, sizeof(double) * batch_size * time_step * map_size);
		neuron_backup = (double*)realloc(neuron_backup, sizeof(double) * batch_size * time_step * map_size);
		neuron_normalized = (double*)realloc(neuron_normalized, sizeof(double) * batch_size * time_step * map_size);
		this->batch_size = batch_size;
	}
	this->epsilon = epsilon;
}

void Neural_Networks::Activate(char option[], int layer_index, int map_index, int time_index) {
	int i = layer_index;
	int j = map_index;
	int t = time_index;

	map_index *= map_size[i];

	if (type_layer[i][0] == 'C') {
		if (strstr(type_layer[i], "lstm")) {
			if (strstr(type_layer[i], "bn")) {
				for (int h = 0; h < LSTM::number_node_types - 1; h++) {
					lstm_batch_normalization[h][0][i][j].Activate(option, number_nodes[i], t, &lstm_neuron[h][0][i][map_index]);
					lstm_batch_normalization[h][1][i][j].Activate(option, number_nodes[i], t, &lstm_neuron[h][1][i][map_index]);
				}
			}
			for (int h = 0; h < batch_size; h++) {
				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = (h * time_step + t) * number_nodes[i] + k;

					lstm_neuron[input][0][i][index] += lstm_neuron[input][1][i][index] + lstm_neuron[input][2][i][index];
					lstm_neuron[input][0][i][index] = 1 / (1 + exp(-lstm_neuron[input][0][i][index]));

					lstm_neuron[forget][0][i][index] += lstm_neuron[forget][1][i][index] + lstm_neuron[forget][2][i][index];
					lstm_neuron[forget][0][i][index] = 1 / (1 + exp(-lstm_neuron[forget][0][i][index]));

					lstm_neuron[output][0][i][index] += lstm_neuron[output][1][i][index] + lstm_neuron[output][2][i][index];
					lstm_neuron[output][0][i][index] = 1 / (1 + exp(-lstm_neuron[output][0][i][index]));

					lstm_neuron[cell][0][i][index] += lstm_neuron[cell][1][i][index] + lstm_neuron[cell][2][i][index];
					lstm_neuron[cell][0][i][index] = 2 / (1 + exp(-2 * lstm_neuron[cell][0][i][index])) - 1;
				}
			}
		}
		else {
			if (strstr(type_layer[i], "bn")) {
				if (neuron[0][i]) batch_normalization[0][i][j].Activate(option, number_nodes[i], t, &neuron[0][i][map_index]);
				if (neuron[1][i]) batch_normalization[1][i][j].Activate(option, number_nodes[i], t, &neuron[1][i][map_index]);
			}
			for (int h = 0; h < batch_size; h++) {
				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = (h * time_step + t) * number_nodes[i] + k;

					double &neuron = this->neuron[0][i][index];

					if (this->neuron[1][i]) neuron += this->neuron[1][i][index];
					if (this->neuron[2][i]) neuron += this->neuron[2][i][index];

					if (strstr(type_layer[i], "ht") || strstr(type_layer[i], "rc")) {
						neuron = 2 / (1 + exp(-2 * neuron)) - 1;
					}
					else if (strstr(type_layer[i], "ls")) {
						neuron = 1 / (1 + exp(-neuron));
					}
					else {
						neuron *= (neuron > 0);
					}
				}
			}
		}
	}
	else if (type_layer[i][0] == 'L') {
		for (int h = 0; h < batch_size; h++) {
			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double &neuron = this->neuron[0][i][(h * time_step + t) * number_nodes[i] + k];

				if (strstr(type_layer[i], "ce")) {
					if (strstr(type_layer[i], "sm")) {
						// neuron = neuron;
					}
					else {
						neuron = 1 / (1 + exp(-neuron));
					}
				}
				else if (strstr(type_layer[i], "mse")) {
					if (strstr(type_layer[i], "ht")) {
						neuron = 2 / (1 + exp(-2 * neuron)) - 1;
					}
					else if (strstr(type_layer[i], "ia")) {
						// neuron = neuron;
					}
					else {
						neuron = 1 / (1 + exp(-neuron));
					}
				}
			}
		}
	}
}
void Neural_Networks::Adjust_Parameter(int layer_index, int map_index) {
	int i = layer_index;
	int j = map_index;

	map_index *= map_size[i];

	if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L') {
		if (strstr(type_layer[i], "lstm")) {
			if (strstr(type_layer[i], "bn")) {
				for (int h = 0; h < LSTM::number_node_types - 1; h++) {
					lstm_batch_normalization[h][0][i][j].Adjust_Parameter();
					lstm_batch_normalization[h][1][i][j].Adjust_Parameter();
				}
				lstm_batch_normalization[cell_output][0][i][j].Adjust_Parameter();
			}
			for (int h = 0; h < batch_size * time_step; h++) {
				int index = h * number_nodes[i];

				double *input_derivative = &lstm_derivative[input][1][i][index];
				double *forget_derivative = &lstm_derivative[forget][1][i][index];
				double *output_derivative = &lstm_derivative[output][1][i][index];
				double *cell_derivative = &lstm_derivative[cell][1][i][index];

				double *previous_neuron = (h % time_step) ? (&neuron[0][i][(h - 1) * number_nodes[i]]) : (nullptr);

				if (previous_neuron) {
					for (int k = map_index; k < map_index + map_size[i]; k++) {
						for (int l = 0; l < number_nodes[i]; l++) {
							int index = k * number_nodes[i] + l;

							lstm_recurrent_weight[input][i][index] -= input_derivative[k] * previous_neuron[l];
							lstm_recurrent_weight[forget][i][index] -= forget_derivative[k] * previous_neuron[l];
							lstm_recurrent_weight[output][i][index] -= output_derivative[k] * previous_neuron[l];
							recurrent_weight[i][index] -= cell_derivative[k] * previous_neuron[l];
						}
					}
				}
			}
			for (int h = 0; h < batch_size * time_step; h++) {
				int index = h * number_nodes[i];

				double *input_derivative = &lstm_derivative[input][0][i][index];
				double *forget_derivative = &lstm_derivative[forget][0][i][index];
				double *output_derivative = &lstm_derivative[output][0][i][index];
				double *cell_derivative = &lstm_derivative[cell][0][i][index];

				double *lower_neuron = &neuron[0][i - 1][h * number_nodes[i - 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					vector<connection> &from_neuron = this->from_neuron[0][i][k];

					for (auto connection = from_neuron.begin(); connection->lower_node != -1; connection++) {
						lstm_weight[input][i][connection->weight] -= input_derivative[k] * lower_neuron[connection->lower_node];
						lstm_weight[forget][i][connection->weight] -= forget_derivative[k] * lower_neuron[connection->lower_node];
						lstm_weight[output][i][connection->weight] -= output_derivative[k] * lower_neuron[connection->lower_node];
						weight[i][connection->weight] -= cell_derivative[k] * lower_neuron[connection->lower_node];
					}
					lstm_weight[input][i][from_neuron.back().weight] -= lstm_derivative[input][2][i][index + k];
					lstm_weight[forget][i][from_neuron.back().weight] -= lstm_derivative[forget][2][i][index + k];
					lstm_weight[output][i][from_neuron.back().weight] -= lstm_derivative[output][2][i][index + k];
					weight[i][from_neuron.back().weight] -= lstm_derivative[cell][2][i][index + k];
				}
			}
		}
		else{
			if (strstr(type_layer[i], "bn")) {
				if (neuron[0][i]) batch_normalization[0][i][j].Adjust_Parameter();
				if (neuron[1][i]) batch_normalization[1][i][j].Adjust_Parameter();
			}
			if (strstr(type_layer[i], "rc")) {
				for (int h = 0; h < batch_size * time_step; h++) {
					double *derivative = &this->derivative[1][i][h * number_nodes[i]];
					double *previous_neuron = (h % time_step) ? (&this->neuron[0][i][(h - 1) * number_nodes[i]]) : (nullptr);

					if (previous_neuron) {
						for (int k = map_index; k < map_index + map_size[i]; k++) {
							for (int l = 0; l < number_nodes[i]; l++) {
								recurrent_weight[i][k * number_nodes[i] + l] -= derivative[k] * previous_neuron[l];
							}
						}
					}
				}
			}

			for (int h = 0; h < batch_size * time_step; h++) {
				double *derivative = &this->derivative[0][i][h * number_nodes[i]];
				double *lower_neuron = &this->neuron[0][i - 1][h * number_nodes[i - 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					vector<connection> &from_neuron = this->from_neuron[0][i][k];

					for (auto connection = from_neuron.begin(); connection->lower_node != -1; connection++) {
						weight[i][connection->weight] -= derivative[k] * lower_neuron[connection->lower_node];
					}
					weight[i][from_neuron.back().weight] -= ((this->derivative[2][i]) ? (this->derivative[2][i][h * number_nodes[i] + k]) : (derivative[k]));
				}
			}
		}
	}
}
void Neural_Networks::Backpropagate(int layer_index, int map_index) {
	if (layer_index == number_layers - 1) {
		return;
	}

	int i = layer_index;
	int j = map_index;

	map_index *= map_size[i];

	if (type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L') {
		if (strstr(type_layer[i + 1], "lstm")) {
			for (int h = 0; h < batch_size * time_step; h++) {
				double *upper_input_derivative = &lstm_derivative[input][0][i + 1][h * number_nodes[i + 1]];
				double *upper_forget_derivative = &lstm_derivative[forget][0][i + 1][h * number_nodes[i + 1]];
				double *upper_output_derivative = &lstm_derivative[output][0][i + 1][h * number_nodes[i + 1]];
				double *upper_cell_derivative = &lstm_derivative[cell][0][i + 1][h * number_nodes[i + 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					double sum = 0;

					vector<connection> &from_derivative = this->from_derivative[0][i][k];

					for (auto connection = from_derivative.begin(); connection != from_derivative.end(); connection++) {
						sum += upper_input_derivative[connection->upper_node] * lstm_weight[input][i + 1][connection->weight];
						sum += upper_forget_derivative[connection->upper_node] * lstm_weight[forget][i + 1][connection->weight];
						sum += upper_output_derivative[connection->upper_node] * lstm_weight[output][i + 1][connection->weight];
						sum += upper_cell_derivative[connection->upper_node] * weight[i + 1][connection->weight];
					}
					derivative[0][i][h * number_nodes[i] + k] = sum;
				}
			}
		}
		else {
			for (int h = 0; h < batch_size * time_step; h++) {
				double *upper_derivative = &this->derivative[0][i + 1][h * number_nodes[i + 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					double sum = 0;

					vector<connection> &from_derivative = this->from_derivative[0][i][k];

					for (auto connection = from_derivative.begin(); connection != from_derivative.end(); connection++) {
						sum += upper_derivative[connection->upper_node] * weight[i + 1][connection->weight];
					}
					derivative[0][i][h * number_nodes[i] + k] = sum;
				}
			}
		}
	}
	else if (type_layer[i + 1][0] == 'P') {
		for (int h = 0; h < batch_size * time_step; h++) {
			double *upper_derivative = &this->derivative[0][i + 1][h * number_nodes[i + 1]];

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				derivative[0][i][h * number_nodes[i] + k] = upper_derivative[from_derivative[0][i][k].begin()->upper_node];
			}
		}
	}
}
void Neural_Networks::Differentiate(int layer_index, int map_index, int time_index, bool output_mask[], double learning_rate, double target_output[]) {
	int i = layer_index;
	int j = map_index;
	int t = time_index;

	map_index *= map_size[i];

	if (type_layer[i][0] == 'C') {
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < batch_size; h++) {
				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = (h * time_step + t) * number_nodes[i] + k;

					lstm_derivative[input][0][i][index] *= (1 - lstm_neuron[input][0][i][index]) * lstm_neuron[input][0][i][index];
					lstm_derivative[input][1][i][index] = lstm_derivative[input][0][i][index];
					lstm_derivative[input][2][i][index] = lstm_derivative[input][0][i][index];

					lstm_derivative[forget][0][i][index] *= (1 - lstm_neuron[forget][0][i][index]) * lstm_neuron[forget][0][i][index];
					lstm_derivative[forget][1][i][index] = lstm_derivative[forget][0][i][index];
					lstm_derivative[forget][2][i][index] = lstm_derivative[forget][0][i][index];

					lstm_derivative[output][0][i][index] *= (1 - lstm_neuron[output][0][i][index]) * lstm_neuron[output][0][i][index];
					lstm_derivative[output][1][i][index] = lstm_derivative[output][0][i][index];
					lstm_derivative[output][2][i][index] = lstm_derivative[output][0][i][index];

					lstm_derivative[cell][0][i][index] *= (1 - lstm_neuron[cell][0][i][index]) * (1 + lstm_neuron[cell][0][i][index]);
					lstm_derivative[cell][1][i][index] = lstm_derivative[cell][0][i][index];
					lstm_derivative[cell][2][i][index] = lstm_derivative[cell][0][i][index];
				}
			}
			if (strstr(type_layer[i], "bn")) {
				for (int h = 0; h < LSTM::number_node_types - 1; h++) {
					lstm_batch_normalization[h][0][i][j].Differentiate(number_nodes[i], t, &lstm_derivative[h][0][i][map_index]);
					lstm_batch_normalization[h][1][i][j].Differentiate(number_nodes[i], t, &lstm_derivative[h][1][i][map_index]);
				}
			}
		}
		else {
			for (int h = 0; h < batch_size; h++) {
				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = (h * time_step + t) * number_nodes[i] + k;

					double &derivative = this->derivative[0][i][index];
					double &neuron = this->neuron[0][i][index];

					if (strstr(type_layer[i], "ht") || strstr(type_layer[i], "rc")) {
						derivative *= (1 - neuron) * (1 + neuron);
					}
					else if (strstr(type_layer[i], "ls")) {
						derivative *= (1 - neuron) * neuron;
					}
					else {
						derivative *= (neuron > 0);
					}

					if (this->derivative[1][i]) this->derivative[1][i][index] = derivative;
					if (this->derivative[2][i]) this->derivative[2][i][index] = derivative;
				}
			}
			if (strstr(type_layer[i], "bn")) {
				if (derivative[0][i]) batch_normalization[0][i][j].Differentiate(number_nodes[i], t, &derivative[0][i][map_index]);
				if (derivative[1][i]) batch_normalization[1][i][j].Differentiate(number_nodes[i], t, &derivative[1][i][map_index]);
			}
		}
	}
	else if (type_layer[i][0] == 'L') {
		for (int h = 0; h < batch_size; h++) {
			for (int k = map_index; k < map_index + map_size[i]; k++) {
				int index = (h * time_step + t) * number_nodes[i] + k;

				double &derivative = this->derivative[0][i][index];
				double &neuron = this->neuron[0][i][index];

				if (output_mask == nullptr || output_mask[t]) {
					derivative = learning_rate * (neuron - target_output[index]);

					if (strstr(type_layer[i], "ce")) {
						if (strstr(type_layer[i], "sm")) {
							// derivative = derivative;
						}
						else {
							// derivative = derivative;
						}
					}
					else if (strstr(type_layer[i], "mse")) {
						if (strstr(type_layer[i], "ht")) {
							derivative *= (1 - neuron) * (1 + neuron);
						}
						else if (strstr(type_layer[i], "ia")) {
							// derivative *= 1;
						}
						else {
							derivative *= (1 - neuron) * neuron;
						}
					}
				}
				else {
					derivative = 0;
				}
			}
		}
	}
}
void Neural_Networks::Feedforward(int layer_index, int map_index) {
	int i = layer_index;
	int j = map_index;

	map_index *= map_size[i];

	if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L') {
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < batch_size * time_step; h++) {
				double *lower_neuron = &this->neuron[0][i - 1][h * number_nodes[i - 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = h * number_nodes[i] + k;

					double sum[LSTM::number_node_types] = { 0, };

					vector<connection> &from_neuron = this->from_neuron[0][i][k];

					for (auto connection = from_neuron.begin(); connection->lower_node != -1; connection++) {
						sum[input] += lower_neuron[connection->lower_node] * lstm_weight[input][i][connection->weight];
						sum[forget] += lower_neuron[connection->lower_node] * lstm_weight[forget][i][connection->weight];
						sum[output] += lower_neuron[connection->lower_node] * lstm_weight[output][i][connection->weight];
						sum[cell] += lower_neuron[connection->lower_node] * weight[i][connection->weight];
					}
					lstm_neuron[input][0][i][index] = sum[input];
					lstm_neuron[forget][0][i][index] = sum[forget];
					lstm_neuron[output][0][i][index] = sum[output];
					lstm_neuron[cell][0][i][index] = sum[cell];

					lstm_neuron[input][2][i][index] = lstm_weight[input][i][from_neuron.back().weight];
					lstm_neuron[forget][2][i][index] = lstm_weight[forget][i][from_neuron.back().weight];
					lstm_neuron[output][2][i][index] = lstm_weight[output][i][from_neuron.back().weight];
					lstm_neuron[cell][2][i][index] = weight[i][from_neuron.back().weight];
				}
			}
		}
		else {
			for (int h = 0; h < batch_size * time_step; h++) {
				double *lower_neuron = &this->neuron[0][i - 1][h * number_nodes[i - 1]];

				for (int k = map_index; k < map_index + map_size[i]; k++) {
					int index = h * number_nodes[i] + k;

					double sum = 0;

					vector<connection> &from_neuron = this->from_neuron[0][i][k];

					for (auto connection = from_neuron.begin(); connection->lower_node != -1; connection++) {
						sum += lower_neuron[connection->lower_node] * weight[i][connection->weight];
					}
					neuron[0][i][index] = sum;

					if (neuron[2][i]) {
						neuron[2][i][index] = weight[i][from_neuron.back().weight];
					}
					else {
						neuron[0][i][index] += weight[i][from_neuron.back().weight];
					}
				}
			}
		}
	}
	else if (type_layer[i][0] == 'P') {
		for (int h = 0; h < batch_size * time_step; h++) {
			double *lower_neuron = &this->neuron[0][i - 1][h * number_nodes[i - 1]];
			double *neuron = &this->neuron[0][i][h * number_nodes[i]];

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				vector<connection> &from_neuron = this->from_neuron[0][i][k];

				if (strstr(type_layer[i], "avg")) {
					double sum = 0;

					for (auto connection = from_neuron.begin(); connection != from_neuron.end(); connection++) {
						sum += lower_neuron[connection->lower_node];
					}
					neuron[k] = sum / from_neuron.size();
				}
				else if (strstr(type_layer[i], "max")) {
					double max = -1;

					for (auto connection = from_neuron.begin(); connection != from_neuron.end(); connection++) {
						if (max < lower_neuron[connection->lower_node]) {
							max = lower_neuron[connection->lower_node];
						}
					}
					neuron[k] = max;
				}
			}
		}
	}
}
void Neural_Networks::LSTM_Backward(int layer_index, int map_index, int time_index) {
	if (strstr(type_layer[layer_index], "lstm")) {
		int i = layer_index;
		int j = map_index;
		int t = time_index;

		map_index *= map_size[i];

		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *output_derivative = &lstm_derivative[output][0][i][index];
			double *cell_output_derivative = &lstm_derivative[cell_output][0][i][index];

			double *output_neuron = &lstm_neuron[output][0][i][index];
			double *cell_output_neuron = &lstm_neuron[cell_output][0][i][index];

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double active_cell_output_neuron = 2 / (1 + exp(-2 * cell_output_neuron[k])) - 1;

				output_derivative[k] = derivative[0][i][index + k] * active_cell_output_neuron;
				cell_output_derivative[k] = derivative[0][i][index + k] * output_neuron[k] * (1 - active_cell_output_neuron) * (1 + active_cell_output_neuron);
			}
		}
		if (strstr(type_layer[i], "bn")) {
			lstm_batch_normalization[cell_output][0][i][j].Differentiate(number_nodes[i], t, &lstm_derivative[cell_output][0][i][map_index]);
		}
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *input_derivative = &lstm_derivative[input][0][i][index];
			double *forget_derivative = &lstm_derivative[forget][0][i][index];
			double *cell_derivative = &lstm_derivative[cell][0][i][index];
			double *cell_output_derivative = &lstm_derivative[cell_output][0][i][index];

			double *input_neuron = &lstm_neuron[input][0][i][index];
			double *forget_neuron = &lstm_neuron[forget][0][i][index];
			double *cell_neuron = &lstm_neuron[cell][0][i][index];

			double *previous_cell_output_neuron = (t) ? ((lstm_batch_normalization[cell_output][0]) ? (&lstm_batch_normalization[cell_output][0][i][j].neuron_backup[(h * time_step + t - 1) * map_size[i]]) : (&lstm_neuron[cell_output][0][i][index - number_nodes[i] + map_index])) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				cell_output_derivative[k] += ((t + 1 < time_step) ? (cell_output_derivative[k + number_nodes[i]] * forget_neuron[k + number_nodes[i]]) : (0));

				input_derivative[k] = cell_output_derivative[k] * cell_neuron[k];
				forget_derivative[k] = (t) ? (cell_output_derivative[k] * previous_cell_output_neuron[k - map_index]) : (0);
				cell_derivative[k] = cell_output_derivative[k] * input_neuron[k];
			}
		}
	}
}
void Neural_Networks::LSTM_Forward(string option, int layer_index, int map_index, int time_index) {
	if (strstr(type_layer[layer_index], "lstm")) {
		int i = layer_index;
		int j = map_index;
		int t = time_index;

		map_index *= map_size[i];
				
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *input_neuron = &lstm_neuron[input][0][i][index];
			double *forget_neuron = &lstm_neuron[forget][0][i][index];
			double *cell_neuron = &lstm_neuron[cell][0][i][index];
			double *cell_output_neuron = &lstm_neuron[cell_output][0][i][index];

			double *previous_cell_output_neuron = (t) ? ((lstm_batch_normalization[cell_output][0]) ? (&lstm_batch_normalization[cell_output][0][i][j].neuron_backup[(h * time_step + t - 1) * map_size[i]]) : (&lstm_neuron[cell_output][0][i][index - number_nodes[i] + map_index])) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				cell_output_neuron[k] = ((t) ? (forget_neuron[k] * previous_cell_output_neuron[k - map_index]) : (0)) + input_neuron[k] * cell_neuron[k];
			}
		}
		if (strstr(type_layer[i], "bn")) {
			lstm_batch_normalization[cell_output][0][i][j].Activate(option, number_nodes[i], t, &lstm_neuron[cell_output][0][i][map_index]);
		}
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *output_neuron = &lstm_neuron[output][0][i][index];
			double *cell_output_neuron = &lstm_neuron[cell_output][0][i][index];

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				neuron[0][i][index + k] = output_neuron[k] * (2 / (1 + exp(-2 * cell_output_neuron[k])) - 1);
			}
		}
	}
}
void Neural_Networks::Recurrent_Backward(int layer_index, int map_index, int time_index) {
	int i = layer_index;
	int j = map_index;
	int t = time_index;

	map_index *= map_size[i];

	if (strstr(type_layer[i], "lstm")) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *next_input_derivative = (t + 1 < time_step) ? (&lstm_derivative[input][1][i][index + number_nodes[i]]) : (nullptr);
			double *next_forget_derivative = (t + 1 < time_step) ? (&lstm_derivative[forget][1][i][index + number_nodes[i]]) : (nullptr);
			double *next_output_derivative = (t + 1 < time_step) ? (&lstm_derivative[output][1][i][index + number_nodes[i]]) : (nullptr);
			double *next_cell_derivative = (t + 1 < time_step) ? (&lstm_derivative[cell][1][i][index + number_nodes[i]]) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double sum = 0;

				if (t + 1 < time_step) {
					for (int l = 0; l < number_nodes[i]; l++) {
						int index = l * number_nodes[i] + k;

						sum += next_input_derivative[l] * lstm_recurrent_weight[input][i][index];
						sum += next_forget_derivative[l] * lstm_recurrent_weight[forget][i][index];
						sum += next_output_derivative[l] * lstm_recurrent_weight[output][i][index];
						sum += next_cell_derivative[l] * recurrent_weight[i][index];
					}
				}
				derivative[0][i][index + k] += sum;
			}
		}
	}
	if (strstr(type_layer[i], "rc")) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *next_derivative = (t + 1 < time_step) ? (&this->derivative[1][i][index + number_nodes[i]]) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double sum = 0;

				if (next_derivative) {
					for (int l = 0; l < number_nodes[i]; l++) {
						sum += next_derivative[l] * recurrent_weight[i][l * number_nodes[i] + k];
					}
				}
				derivative[0][i][index + k] += sum;
			}
		}
	}
}
void Neural_Networks::Recurrent_Forward(int layer_index, int map_index, int time_index) {
	int i = layer_index;
	int j = map_index;
	int t = time_index;

	map_index *= map_size[i];

	if (strstr(type_layer[i], "lstm")) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *previous_neuron = (t) ? (&neuron[0][i][index - number_nodes[i]]) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double sum[LSTM::number_node_types] = { 0, };

				if (previous_neuron) {
					for (int l = 0; l < number_nodes[i]; l++) {
						int index = k * number_nodes[i] + l;

						sum[input] += previous_neuron[l] * lstm_recurrent_weight[input][i][index];
						sum[forget] += previous_neuron[l] * lstm_recurrent_weight[forget][i][index];
						sum[output] += previous_neuron[l] * lstm_recurrent_weight[output][i][index];
						sum[cell] += previous_neuron[l] * recurrent_weight[i][index];
					}
				}
				lstm_neuron[input][1][i][index + k] = sum[input];
				lstm_neuron[forget][1][i][index + k] = sum[forget];
				lstm_neuron[output][1][i][index + k] = sum[output];
				lstm_neuron[cell][1][i][index + k] = sum[cell];
			}
		}
	}
	if (strstr(type_layer[i], "rc")) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * number_nodes[i];

			double *previous_neuron = (t) ? (&neuron[0][i][index - number_nodes[i]]) : (nullptr);

			for (int k = map_index; k < map_index + map_size[i]; k++) {
				double sum = 0;

				if (previous_neuron) {
					for (int l = 0; l < number_nodes[i]; l++) {
						sum += previous_neuron[l] * recurrent_weight[i][k * number_nodes[i] + l];
					}
				}
				neuron[1][i][index + k] = sum;
			}
		}
	}
}
void Neural_Networks::Softmax(int layer_index) {
	if (strstr(type_layer[layer_index], "sm")) {
		int i = layer_index;

		for (int h = 0; h < batch_size * time_step; h++) {
			double max = 0;
			double sum = 0;

			double *neuron = &this->neuron[0][i][h * number_nodes[i]];

			for (int j = 0; j < number_nodes[i]; j++) {
				if (max < neuron[j]) {
					max = neuron[j];
				}
			}
			for (int j = 0; j < number_nodes[i]; j++) {
				neuron[j] = exp(neuron[j] - max);
				sum += neuron[j];
			}
			for (int j = 0; j < number_nodes[i]; j++) {
				neuron[j] /= sum;
			}
		}
	}
}

void Neural_Networks::Construct_Networks() {
	unordered_map<int, int> *weight_index = new unordered_map<int, int>[number_layers];

	kernel_width = new int[number_layers];
	kernel_height = new int[number_layers];
	map_size = new int[number_layers];
	number_nodes = new int[number_layers];
	number_weights = new int[number_layers];
	stride_width = new int[number_layers];
	stride_height = new int[number_layers];

	for (int i = 0; i < number_layers; i++) {
		map_size[i] = map_height[i] * map_width[i];
		number_nodes[i] = number_maps[i] * map_height[i] * map_width[i];

		if (strstr(type_layer[i], "ks")) {
			char *kernel_size = strstr(type_layer[i], "ks");

			kernel_width[i] = atoi(kernel_size + 2);
			kernel_size = strstr(kernel_size + 2, ",");
			kernel_height[i] = (kernel_size && atoi(kernel_size + 1) > 0) ? (atoi(kernel_size + 1)) : (kernel_width[i]);
		}
		else {
			kernel_width[i] = (i == 0 || type_layer[i][0] == 'P') ? (0) : (abs(map_width[i - 1] - map_width[i]) + 1);
			kernel_height[i] = (i == 0 || type_layer[i][0] == 'P') ? (0) : (abs(map_height[i - 1] - map_height[i]) + 1);
		}

		if (strstr(this->type_layer[i], ",st")) {
			char *stride = strstr(type_layer[i], ",st");

			stride_width[i] = atoi(stride + 2);
			stride = strstr(stride + 2, ",");
			stride_height[i] = (stride && atoi(stride + 1) > 0) ? (atoi(stride + 1)) : (stride_width[i]);
		}
		else if(type_layer[i][0] == 'P'){
			stride_width[i] = (map_width[i - 1] > map_width[i]) ? (map_width[i - 1] / map_width[i]) : (map_width[i] / map_width[i - 1]);
			stride_height[i] = (map_height[i - 1] > map_height[i]) ? (map_height[i - 1] / map_height[i]) : (map_height[i] / map_height[i - 1]);
		}
		else {
			stride_width[i] = 1;
			stride_height[i] = 1;
		}
	}

	for (int i = 0; i < number_layers; i++) {
		if (kernel_width[i]) {
			bool depthwise_separable = (strstr(type_layer[i], "dw") != 0);

			number_weights[i] = 0;

			for (int j = 0, index = 0; j < number_maps[i]; j++) {
				for (int k = 0; k < number_maps[i - 1] + 1; k++) {
					if (!depthwise_separable || j == k || k == number_maps[i - 1]) {
						for (int l = 0; l < kernel_height[i] * kernel_width[i]; l++) {
							weight_index[i].insert(pair<int, int>(j * (number_maps[i - 1] + 1) * kernel_height[i] * kernel_width[i] + k * kernel_height[i] * kernel_width[i] + l, index++));
						}
						number_weights[i] += kernel_height[i] * kernel_width[i];
					}
				}
			}
			if (weight == nullptr) {
				weight = new double*[number_layers];
			}
			weight[i] = new double[number_weights[i]];
		}
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < LSTM::number_weight_types; h++) {
				if (lstm_recurrent_weight[h] == nullptr) lstm_recurrent_weight[h] = new double*[number_layers];
				if (lstm_weight[h] == nullptr) lstm_weight[h] = new double*[number_layers];

				lstm_recurrent_weight[h][i] = new double[number_nodes[i] * number_nodes[i]];
				lstm_weight[h][i] = new double[number_weights[i]];
			}
		}
		if (strstr(type_layer[i], "rc") || strstr(type_layer[i], "lstm")) {
			if (recurrent_weight == nullptr) {
				recurrent_weight = new double*[number_layers];
			}
			recurrent_weight[i] = new double[number_nodes[i] * number_nodes[i]];
		}
	}

	for (int h = 0; h < number_batch_types; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (strstr(type_layer[i], "bn") && Access_Node(h, i)) {
				if (batch_normalization[h] == nullptr) {
					batch_normalization[h] = new Batch_Normalization*[number_layers];
				}
				batch_normalization[h][i] = new Batch_Normalization[number_maps[i]];

				for (int j = 0; j < number_maps[i]; j++) {
					batch_normalization[h][i][j].Construct(batch_size, map_size[i], time_step);
				}

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						if (lstm_batch_normalization[g][h] == nullptr) lstm_batch_normalization[g][h] = new Batch_Normalization*[number_layers];

						lstm_batch_normalization[g][h][i] = new Batch_Normalization[number_maps[i]];

						for (int j = 0; j < number_maps[i]; j++) {
							lstm_batch_normalization[g][h][i][j].Construct(batch_size, map_size[i], time_step);
						}
					}
				}
			}
		}
	}

	for (int h = 0; h < number_node_types; h++) {
		memset(derivative[h] = new double*[number_layers], 0, sizeof(double*) * number_layers);
		memset(neuron[h] = new double*[number_layers], 0, sizeof(double*) * number_layers);

		for (int i = 0; i < number_layers; i++) {
			if (Access_Node(h, i)) {
				derivative[h][i] = new double[batch_size * time_step * number_nodes[i]];
				neuron[h][i] = new double[batch_size * time_step * number_nodes[i]];

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						if (lstm_derivative[g][h] == nullptr) lstm_derivative[g][h] = new double*[number_layers];
						if (lstm_neuron[g][h] == nullptr) lstm_neuron[g][h] = new double*[number_layers];

						lstm_derivative[g][h][i] = new double[batch_size * time_step * number_nodes[i]];
						lstm_neuron[g][h][i] = new double[batch_size * time_step * number_nodes[i]];
					}
				}
			}
		}
	}

	for (int h = 0; h < 3; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (Access_Node(h, i)) {
				if(from_derivative[h] == nullptr) from_derivative[h] = new vector<connection>*[number_layers];
				if(from_neuron[h] == nullptr) from_neuron[h] = new vector<connection>*[number_layers];

				from_derivative[h][i] = new vector<connection>[number_nodes[i]];
				from_neuron[h][i] = new vector<connection>[number_nodes[i]];
			}
		}
	}

	for (int i = 1; i < number_layers; i++) {
		bool depthwise_separable = (strstr(type_layer[i], "dw") != 0);

		for (int j = 0; j < number_maps[i]; j++) {
			for (int k = 0; k < map_height[i]; k++) {
				for (int l = 0; l < map_width[i]; l++) {
					int index[2] = { j * map_size[i] + k * map_width[i] + l, 0 };

					connection connection;

					if (type_layer[i][0] == 'C' || type_layer[i][0] == 'L') {
						for (int m = 0; m < number_maps[i - 1]; m++) {
							if (!depthwise_separable || j == m) {
								for (int n = 0; n < map_height[i - 1]; n++) {
									for (int o = 0; o < map_width[i - 1]; o++) {
										int distance[2] = { (map_height[i] < map_height[i - 1]) ? (n - k * stride_height[i]) : (k - n * stride_height[i]) , (map_width[i] < map_width[i - 1]) ? (o - l * stride_width[i]) : (l - o * stride_width[i]) };

										if (0 <= distance[0] && distance[0] < kernel_height[i] && 0 <= distance[1] && distance[1] < kernel_width[i]) {
											index[1] = m * map_size[i - 1] + n * map_width[i - 1] + o;

											connection.lower_node = index[1];
											connection.upper_node = index[0];
											connection.weight = weight_index[i].find(j * (number_maps[i - 1] + 1) * kernel_height[i] * kernel_width[i] + m * kernel_height[i] * kernel_width[i] + distance[0] * kernel_width[i] + distance[1])->second;

											from_derivative[0][i - 1][index[1]].push_back(connection);
											from_neuron[0][i][index[0]].push_back(connection);
										}
									}
								}
							}
						}
						connection.lower_node = -1;
						connection.weight = weight_index[i].find(j * (number_maps[i - 1] + 1) * kernel_height[i] * kernel_width[i] + number_maps[i - 1] * kernel_height[i] * kernel_width[i])->second;

						from_neuron[0][i][index[0]].push_back(connection);
					}
					else if (type_layer[i][0] == 'P') {
						for (int n = 0; n < map_height[i - 1]; n++) {
							for (int o = 0; o < map_width[i - 1]; o++) {
								int distance[2] = { (map_height[i] < map_height[i - 1]) ? (n - k * stride_height[i]) : (k - n * stride_height[i]) , (map_width[i] < map_width[i - 1]) ? (o - l * stride_width[i]) : (l - o * stride_width[i]) };

								if (0 <= distance[0] && distance[0] < ((kernel_height[i]) ? (kernel_height[i]) : (stride_height[i])) && 0 <= distance[1] && distance[1] < ((kernel_width[i]) ? (kernel_width[i]) : (stride_width[i]))) {
									index[1] = j * map_height[i - 1] * map_width[i - 1] + n * map_width[i - 1] + o;

									connection.lower_node = index[1];
									connection.upper_node = index[0];

									from_derivative[0][i - 1][index[1]].push_back(connection);
									from_neuron[0][i][index[0]].push_back(connection);
								}
							}
						}
					}
				}
			}
		}
	}
	delete[] weight_index;
}
void Neural_Networks::Resize_Node(int batch_size) {
	if (this->batch_size != batch_size) {
		for (int h = 0; h < number_batch_types; h++) {
			for (int i = 0; i < number_layers; i++) {
				if (strstr(type_layer[i], "bn") && Access_Node(h, i)) {
					for (int j = 0; j < number_maps[i]; j++) {
						batch_normalization[h][i][j].Resize_Node(batch_size, epsilon);
					}

					if (strstr(type_layer[i], "lstm")) {
						for (int g = 0; g < LSTM::number_node_types; g++) {
							for (int j = 0; j < number_maps[i]; j++) {
								lstm_batch_normalization[g][h][i][j].Resize_Node(batch_size, epsilon);
							}
						}
					}
				}
			}
		}
		for (int h = 0; h < number_node_types; h++) {
			for (int i = 0; i < number_layers; i++) {
				if (Access_Node(h, i)) {
					derivative[h][i] = (double*)realloc(derivative[h][i], sizeof(double) * batch_size * time_step * number_nodes[i]);
					neuron[h][i] = (double*)realloc(neuron[h][i], sizeof(double) * batch_size * time_step * number_nodes[i]);

					if (strstr(type_layer[i], "lstm")) {
						for (int g = 0; g < LSTM::number_node_types; g++) {
							lstm_derivative[g][h][i] = (double*)realloc(lstm_derivative[g][h][i], sizeof(double) * batch_size * time_step * number_nodes[i]);
							lstm_neuron[g][h][i] = (double*)realloc(lstm_neuron[g][h][i], sizeof(double) * batch_size * time_step * number_nodes[i]);
						}
					}
				}
			}
		}
		this->batch_size = batch_size;
	}
}

bool Neural_Networks::Access_Node(int type_index, int layer_index) {
	int h = type_index;
	int i = layer_index;

	return (h == 0 || strstr(type_layer[i], "lstm") || strstr(type_layer[i], "rc"));
}

Neural_Networks::Neural_Networks(string path) {
	ifstream file(path);

	if (file.is_open()) {
		file >> number_layers;
		file >> time_step;

		type_layer = new char*[number_layers];
		for (int i = 0; i < number_layers; i++) {
			string type;

			file >> type;
			strcpy(type_layer[i] = new char[type.size() + 1], type.c_str());
		}

		number_maps = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> number_maps[i];
		map_width = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> map_width[i];
		map_height = new int[number_layers];
		for (int i = 0; i < number_layers; i++) file >> map_height[i];
		file >> epsilon;

		Construct_Networks();

		for (int i = 0; i < number_layers; i++) {
			if (strstr(type_layer[i], "bn")) {
				for (int h = 0; h < number_batch_types; h++) {
					if (Access_Node(h, i)) {
						for (int j = 0; j < number_maps[i]; j++) {
							file >> batch_normalization[h][i][j].gamma;
							file >> batch_normalization[h][i][j].beta;

							for (int t = 0; t < time_step; t++) file >> batch_normalization[h][i][j].mean[t];
							for (int t = 0; t < time_step; t++) file >> batch_normalization[h][i][j].variance[t];
						}
					}
				}
			}
			if (kernel_width[i]) {
				for (int j = 0; j < number_weights[i]; j++) {
					file >> weight[i][j];
				}
			}
			if (strstr(type_layer[i], "lstm")) {
				for (int h = 0; h < LSTM::number_weight_types; h++) {
					for (int j = 0; j < number_nodes[i] * number_nodes[i]; j++) {
						file >> lstm_recurrent_weight[h][i][j];
					}
					for (int j = 0; j < number_weights[i]; j++) {
						file >> lstm_weight[h][i][j];
					}
				}
			}
			if (strstr(type_layer[i], "rc") || strstr(type_layer[i], "lstm")) {
				for (int j = 0; j < number_nodes[i] * number_nodes[i]; j++) {
					file >> recurrent_weight[i][j];
				}
			}
		}
		file.close();
	}
	else {
		cerr << "[Convolutional_Neural_Networks], " + path + " not found" << endl;
	}
}
Neural_Networks::Neural_Networks(string type_layer[], int number_layers, int number_maps[], int map_width[], int map_height[], int time_step) {
	this->map_width = new int[number_layers];
	this->map_height = new int[number_layers];
	this->number_layers = number_layers;
	this->number_maps = new int[number_layers];
	this->time_step = time_step;
	this->type_layer = new char*[number_layers];

	for (int i = 0; i < number_layers; i++) {
		strcpy(this->type_layer[i] = new char[type_layer[i].size() + 1], type_layer[i].c_str());
		this->number_maps[i] = number_maps[i];
		this->map_width[i] = (map_width == nullptr) ? (1) : (map_width[i]);
		this->map_height[i] = (map_height == nullptr) ? (1) : (map_height[i]);
	}
	Construct_Networks();
}
Neural_Networks::~Neural_Networks() {
	for (int i = 0; i < number_layers; i++) {
		if (kernel_width[i]) {
			delete[] weight[i];
		}
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < LSTM::number_weight_types; h++) {
				delete[] lstm_recurrent_weight[h][i];
				delete[] lstm_weight[h][i];
			}
		}
		if (strstr(type_layer[i], "rc") || strstr(type_layer[i], "lstm")) {
			delete[] recurrent_weight[i];
		}
	}
	for (int h = 0; h < LSTM::number_weight_types; h++) {
		if (lstm_recurrent_weight[h]) delete[] lstm_recurrent_weight[h];
		if(lstm_weight[h]) delete[] lstm_weight[h];
	}
	if (recurrent_weight) delete[] recurrent_weight;
	if (weight) delete[] weight;

	for (int h = 0; h < number_batch_types; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (strstr(type_layer[i], "bn") && Access_Node(h, i)) {
				delete[] batch_normalization[h][i];

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						delete[] lstm_batch_normalization[g][h][i];
					}
				}
			}
		}
		if (batch_normalization[h]) delete[] batch_normalization[h];

		for (int g = 0; g < LSTM::number_node_types; g++) {
			if(lstm_batch_normalization[g][h]) delete[] lstm_batch_normalization[g][h];
		}
	}

	for (int h = 0; h < number_node_types; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (Access_Node(h, i)) {
				delete[] derivative[h][i];
				delete[] neuron[h][i];

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						delete[] lstm_derivative[g][h][i];
						delete[] lstm_neuron[g][h][i];
					}
				}
			}
		}
		if (derivative[h]) delete[] derivative[h];
		if (neuron[h]) delete[] neuron[h];

		for (int g = 0; g < LSTM::number_node_types; g++) {
			if (lstm_derivative[g][h]) delete[] lstm_derivative[g][h];
			if (lstm_neuron[g][h]) delete[] lstm_neuron[g][h];
		}
	}

	for (int h = 0; h < 3; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (Access_Node(h, i)) {
				delete[] from_derivative[h][i];
				delete[] from_neuron[h][i];
			}
		}
		if (from_derivative[h]) delete[] from_derivative[h];
		if (from_neuron[h]) delete[] from_neuron[h];
	}

	for (int i = 0; i < number_layers; i++) {
		delete[] type_layer[i];
	}
	delete[] type_layer;

	delete[] kernel_width;
	delete[] kernel_height;
	delete[] map_width;
	delete[] map_height;
	delete[] number_maps;
	delete[] number_nodes;
	delete[] stride_width;
	delete[] stride_height;
}

void Neural_Networks::Initialize_Parameter(double mean, double variance, double gamma, int seed) {
	default_random_engine generator(seed);
	normal_distribution<double> distribution(mean, variance);

	for (int i = 0; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int h = 0; h < number_batch_types; h++) {
				if (Access_Node(h, i)) {
					for (int j = 0; j < number_maps[i]; j++) {
						batch_normalization[h][i][j].gamma = gamma;
						batch_normalization[h][i][j].beta = 0;
					}

					if (strstr(type_layer[i], "lstm")) {
						for (int g = 0; g < LSTM::number_node_types; g++) {
							for (int j = 0; j < number_maps[i]; j++) {
								lstm_batch_normalization[g][h][i][j].gamma = gamma;
								lstm_batch_normalization[g][h][i][j].beta = 0;
							}
						}
					}
				}
			}
		}
		if (kernel_width[i]) {
			for (int j = 0; j < number_weights[i]; j++) {
				weight[i][j] = distribution(generator);
			}
		}
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < LSTM::number_weight_types; h++) {
				for (int j = 0; j < number_nodes[i] * number_nodes[i];j++) {
					lstm_recurrent_weight[h][i][j] = distribution(generator);
				}
				for (int j = 0; j < number_weights[i]; j++) {
					lstm_weight[h][i][j] = distribution(generator);
				}
			}
		}
		if (strstr(type_layer[i], "rc") || strstr(type_layer[i], "lstm")) {
			for (int j = 0; j < number_nodes[i] * number_nodes[i]; j++) {
				recurrent_weight[i][j] = distribution(generator);
			}
		}
	}
}
void Neural_Networks::Save_Model(string path) {
	ofstream file(path);

	file << number_layers << endl;
	file << time_step << endl;
	for (int i = 0; i < number_layers; i++) file << type_layer[i] << endl;
	for (int i = 0; i < number_layers; i++) file << number_maps[i] << endl;
	for (int i = 0; i < number_layers; i++) file << map_width[i] << endl;
	for (int i = 0; i < number_layers; i++) file << map_height[i] << endl;
	file << epsilon << endl;

	for (int i = 1; i < number_layers; i++) {
		if (strstr(type_layer[i], "bn")) {
			for (int h = 0; h < number_batch_types; h++) {
				if (Access_Node(h, i)) {
					for (int j = 0; j < number_maps[i]; j++) {
						file << batch_normalization[h][i][j].gamma << endl;
						file << batch_normalization[h][i][j].beta << endl;

						for (int t = 0; t < time_step; t++) file << batch_normalization[h][i][j].mean[t] << endl;
						for (int t = 0; t < time_step; t++) file << batch_normalization[h][i][j].variance[t] << endl;
					}
				}
			}
		}
		if (kernel_width[i]) {
			for (int j = 0; j < number_weights[i]; j++) {
				file << weight[i][j] << endl;
			}
		}
		if (strstr(type_layer[i], "lstm")) {
			for (int h = 0; h < LSTM::number_weight_types; h++) {
				for (int j = 0; j < number_nodes[i] * number_nodes[i]; j++) {
					file << lstm_recurrent_weight[h][i][j] << endl;
				}
				for (int j = 0; j < number_weights[i]; j++) {
					file << lstm_weight[h][i][j] << endl;
				}
			}
		}
		if (strstr(type_layer[i], "rc") || strstr(type_layer[i], "lstm")) {
			for (int j = 0; j < number_nodes[i] * number_nodes[i]; j++) {
				file << recurrent_weight[i][j] << endl;
			}
		}
	}
	file.close();
}
void Neural_Networks::Test(double input[], double output[]) {
	Resize_Node(1);

	memcpy(neuron[0][0], input, sizeof(double) * time_step * number_nodes[0]);

	for (int i = 1; i < number_layers; i++) {
		#pragma omp parallel for
		for (int j = 0; j < number_maps[i]; j++) {
			Feedforward(i, j);
		}
		for (int t = 0; t < time_step; t++) {
			#pragma omp parallel for
			for (int j = 0; j < number_maps[i]; j++) {
				Recurrent_Forward(i, j, t);
				Activate("inference", i, j, t);
				LSTM_Forward("inference", i, j, t);
			}
		}
		Softmax(i);
	}
	memcpy(output, neuron[0][number_layers - 1], sizeof(double) * time_step * number_nodes[number_layers - 1]);
}
void Neural_Networks::Test(int batch_size, double **input, double **output) {
	Resize_Node(batch_size);

	for (int h = 0, i = 0; h < batch_size; h++) {
		memcpy(&neuron[0][i][h * time_step * number_nodes[i]], input[h], sizeof(double) * time_step * number_nodes[i]);
	}
	for (int i = 1; i < number_layers; i++) {
		#pragma omp parallel for
		for (int j = 0; j < number_maps[i]; j++) {
			Feedforward(i, j);
		}
		for (int t = 0; t < time_step; t++) {
			#pragma omp parallel for
			for (int j = 0; j < number_maps[i]; j++) {
				Recurrent_Forward(i, j, t);
				Activate("inference", i, j, t);
				LSTM_Forward("inference", i, j, t);
			}
		}
		Softmax(i);
	}
	for (int h = 0, i = number_layers - 1; h < batch_size; h++) {
		memcpy(output[h], &neuron[0][i][h * time_step * number_nodes[i]], sizeof(double) * time_step * number_nodes[i]);
	}
}

double Neural_Networks::Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output, bool output_mask[], double noise_variance) {
	int *index = new int[number_training];

	double loss = 0;

	double *target_output_batch = new double[batch_size * time_step * number_nodes[number_layers - 1]];

	default_random_engine generator(0);
	normal_distribution<double> distribution(0, noise_variance);

	for (int i = 0; i < number_training; i++) {
		index[i] = i;
	}
	for (int i = 0; i < number_training; i++) {
		int j = rand() % number_training;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;
	}
	this->epsilon = epsilon;
	Resize_Node(batch_size);

	for (int h = 0; h < number_batch_types; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (strstr(type_layer[i], "bn") && Access_Node(h, i)) {
				for (int j = 0; j < number_maps[i]; j++) {
					batch_normalization[h][i][j].Calculate_Mean_Variance();
				}

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						for (int j = 0; j < number_maps[i]; j++) {
							lstm_batch_normalization[g][h][i][j].Calculate_Mean_Variance();
						}
					}
				}
			}
		}
	}

	for (int g = 0, h = 0; g < number_training; g++) {
		memcpy(&neuron[0][0][h * time_step * number_nodes[0]], input[index[g]], sizeof(double) * time_step * number_nodes[0]);
		memcpy(&target_output_batch[h * time_step * number_nodes[number_layers - 1]], target_output[index[g]], sizeof(double) * time_step * number_nodes[number_layers - 1]);

		for (int j = 0, i = 0; j < time_step * number_nodes[i]; j++) {
			neuron[0][i][h * time_step * number_nodes[i] + j] += distribution(generator);
		}

		if (++h == batch_size) {
			h = 0;

			for (int i = 1; i < number_layers; i++) {
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++) {
					Feedforward(i, j);
				}
				for(int t = 0;t < time_step;t++){
					#pragma omp parallel for
					for (int j = 0; j < number_maps[i]; j++) {
						Recurrent_Forward(i, j, t);
						Activate("training", i, j, t);
						LSTM_Forward("training", i, j, t);
					}
				}
				Softmax(i);
			}

			for (int i = number_layers - 1; i > 0; i--) {
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++) {
					Backpropagate(i, j);
				}
				for (int t = time_step - 1; t >= 0; t--) {
					#pragma omp parallel for
					for (int j = 0; j < number_maps[i]; j++) {
						Recurrent_Backward(i, j, t);
						LSTM_Backward(i, j, t);
						Differentiate(i, j, t, output_mask, learning_rate, target_output_batch);
					}
				}
			}
			for (int i = number_layers - 1; i > 0; i--) {
				#pragma omp parallel for
				for (int j = 0; j < number_maps[i]; j++) {
					Adjust_Parameter(i, j);
				}
			}

			for (int h = 0, i = number_layers - 1; h < batch_size * time_step; h++) {
				if (output_mask == nullptr || output_mask[h % time_step]) {
					double *neuron = &this->neuron[0][i][h * number_nodes[i]];
					double *target_output = &target_output_batch[h * number_nodes[i]];

					for (int j = 0; j < number_nodes[i]; j++) {
						if (strstr(type_layer[i], "ce")) {
							loss -= target_output[j] * log(neuron[j] + 0.000001) + (1 - target_output[j]) * log(1 - neuron[j] + 0.000001);
						}
						if (strstr(type_layer[i], "mse")) {
							loss += 0.5 * (neuron[j] - target_output[j]) * (neuron[j] - target_output[j]);
						}
					}
				}
			}
		}
	}


	for (int h = 0; h < number_batch_types; h++) {
		for (int i = 0; i < number_layers; i++) {
			if (strstr(type_layer[i], "bn") && Access_Node(h, i)) {
				for (int j = 0; j < number_maps[i]; j++) {
					batch_normalization[h][i][j].Calculate_Mean_Variance(number_training / batch_size);
				}

				if (strstr(type_layer[i], "lstm")) {
					for (int g = 0; g < LSTM::number_node_types; g++) {
						for (int j = 0; j < number_maps[i]; j++) {
							lstm_batch_normalization[g][h][i][j].Calculate_Mean_Variance(number_training / batch_size);
						}
					}
				}
			}
		}
	}

	delete[] target_output_batch;
	delete[] index;

	return loss / number_training;
}
