#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>

#include "Neural_Networks.h"

Batch_Normalization::Batch_Normalization(int number_maps, int map_size) {
	this->map_size = map_size;
	this->number_maps = number_maps;
	batch_size = 1;
	number_nodes = number_maps * map_size;
	time_step = 1;

	gamma = new float[number_maps];
	gamma_optimizer = new Optimizer();
	gamma_optimizer->Resize_Memory(number_maps);

	beta = new float[number_maps];
	beta_optimizer = new Optimizer();
	beta_optimizer->Resize_Memory(number_maps);

	mean = new float[number_maps];
	variance = new float[number_maps];
	memset(sum_mean = new float[number_maps], 0, sizeof(float) * number_maps);
	memset(sum_variance = new float[number_maps], 0, sizeof(float) * number_maps);

	error_backup = new float[number_nodes];
	error_normalized = new float[number_nodes];
	neuron_backup = new float[number_nodes];
	neuron_normalized = new float[number_nodes];
}
Batch_Normalization::~Batch_Normalization() {
	delete[] gamma;
	delete gamma_optimizer;
	delete[] beta;
	delete beta_optimizer;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;

	delete[] error_backup;
	delete[] error_normalized;
	delete[] neuron_backup;
	delete[] neuron_normalized;
}

void Batch_Normalization::Activate(string phase, float _neuron[], int time_index) {
	int t = time_index;

	if (phase == "training") {
#pragma omp parallel for
		for (int j = 0; j < number_maps; j++) {
			float *mean = &this->mean[t * number_maps];
			float *variance = &this->variance[t * number_maps];
			float *sum_mean = &this->sum_mean[t * number_maps];
			float *sum_variance = &this->sum_variance[t * number_maps];

			float *neuron = &_neuron[t * number_nodes + j * map_size];
			float *neuron_backup = &this->neuron_backup[t * number_nodes + j * map_size];
			float *neuron_normalized = &this->neuron_normalized[t * number_nodes + j * map_size];

			double standard_deviation;
			double sum = 0;

			for (int h = 0; h < batch_size; h++) {
				int index = h * time_step * number_nodes;

				for (int k = 0; k < map_size; k++) {
					sum += neuron[index + k];
				}
			}
			sum_mean[j] += (mean[j] = sum / (batch_size * map_size));

			sum = 0;
			for (int h = 0; h < batch_size; h++) {
				int index = h * time_step * number_nodes;

				for (int k = 0; k < map_size; k++) {
					sum += (neuron[index + k] - mean[j]) * (neuron[index + k] - mean[j]);
				}
			}
			sum_variance[j] += (variance[j] = sum / (batch_size * map_size));
			standard_deviation = sqrt(variance[j] + epsilon);

			for (int h = 0; h < batch_size; h++) {
				int index = h * time_step * number_nodes;

				for (int k = 0; k < map_size; k++) {
					neuron_backup[index + k] = neuron[index + k];
					neuron_normalized[index + k] = (neuron[index + k] - mean[j]) / standard_deviation;
					neuron[index + k] = gamma[j] * neuron_normalized[index + k] + beta[j];
				}
			}
		}
	}
	else if (phase == "inference") {
#pragma omp parallel for
		for (int j = 0; j < number_maps; j++) {
			float *mean = &this->mean[t * number_maps];
			float *neuron = &_neuron[t * number_nodes + j * map_size];
			float *neuron_backup = &this->neuron_backup[t * number_nodes + j * map_size];

			double standard_deviation = sqrt(variance[t * number_maps + j] + epsilon);

			for (int h = 0; h < batch_size; h++) {
				int index = h * time_step * number_nodes;

				for (int k = 0; k < map_size; k++) {
					neuron_backup[index + k] = neuron[index + k];
					neuron[index + k] = gamma[j] / standard_deviation * neuron[index + k] + (beta[j] - gamma[j] * mean[j] / standard_deviation);
				}
			}
		}
	}
}
void Batch_Normalization::Adjust_Parameter(double gradient_clip, double learning_rate) {
#pragma omp parallel for
	for (int j = 0; j < number_maps; j++) {
		gamma[j] += gamma_optimizer->Calculate_Gradient(j, gamma_optimizer->gradient[j], gradient_clip * learning_rate, true);
		beta[j] += beta_optimizer->Calculate_Gradient(j, beta_optimizer->gradient[j], gradient_clip * learning_rate, true);
	}
}
void Batch_Normalization::Calculate_Mean_Variance(int number_batches) {
#pragma omp parallel for
	for (int j = 0; j < time_step * number_maps; j++) {
		mean[j] = sum_mean[j] / number_batches;
		variance[j] = batch_size / (batch_size - 1.0) * sum_variance[j] / number_batches;

		sum_mean[j] = 0;
		sum_variance[j] = 0;
	}
}
void Batch_Normalization::Differentiate(float _error[], int time_index) {
	int t = time_index;

#pragma omp parallel for
	for (int j = 0; j < number_maps; j++) {
		float *mean = &this->mean[t * number_maps];
		float *variance = &this->variance[t * number_maps];

		float *error = &_error[t * number_nodes + j * map_size];
		float *error_backup = &this->error_backup[t * number_nodes + j * map_size];
		float *error_normalized = &this->error_normalized[t * number_nodes + j * map_size];
		float *neuron_backup = &this->neuron_backup[t * number_nodes + j * map_size];

		double error_mean = 0;
		double error_variance = 0;
		double standard_deviation = sqrt(variance[j] + epsilon);

		for (int h = 0; h < batch_size; h++) {
			int index = h * time_step * number_nodes;

			for (int k = 0; k < map_size; k++) {
				error_normalized[index + k] = error[index + k] * gamma[j];
				error_variance += error_normalized[index + k] * (neuron_backup[index + k] - mean[j]);
			}
		}
		error_variance *= (-0.5) * pow(variance[j] + epsilon, (double)-1.5);

		for (int h = 0; h < batch_size; h++) {
			int index = h * time_step * number_nodes;

			for (int k = 0; k < map_size; k++) {
				error_mean += error_normalized[index + k];
			}
		}
		error_mean = -error_mean / standard_deviation;

		for (int h = 0; h < batch_size; h++) {
			int index = h * time_step * number_nodes;

			for (int k = 0; k < map_size; k++) {
				error_backup[index + k] = error[index + k];
				error[index + k] = error_normalized[index + k] / standard_deviation + error_variance * 2 * (neuron_backup[index + k] - mean[j]) / (batch_size * map_size) + error_mean / (batch_size * map_size);
			}
		}
	}
}
void Batch_Normalization::Initialize(double gamma) {
	for (int j = 0; j < number_maps; j++) {
		this->gamma[j] = gamma;
		beta[j] = 0;
	}
	memset(sum_mean, 0, sizeof(float) * time_step * number_maps);
	memset(sum_variance, 0, sizeof(float) * time_step * number_maps);
}
void Batch_Normalization::Load(ifstream &file) {
	for (int j = 0; j < number_maps; j++) file >> gamma[j];
	for (int j = 0; j < number_maps; j++) file >> beta[j];
	for (int j = 0; j < time_step * number_maps; j++) file >> mean[j];
	for (int j = 0; j < time_step * number_maps; j++) file >> variance[j];
}
void Batch_Normalization::Resize_Memory(int batch_size, int time_step) {
	if (this->batch_size != batch_size || this->time_step != time_step) {
		int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

		if (this->time_step != time_step) {
			int memory_size = sizeof(float) * time_step * number_maps;

			mean = (float*)realloc(mean, memory_size);
			variance = (float*)realloc(variance, memory_size);
			memset(sum_mean = (float*)realloc(sum_mean, memory_size), 0, memory_size);
			memset(sum_variance = (float*)realloc(sum_variance, memory_size), 0, memory_size);
		}
		error_backup = (float*)realloc(error_backup, memory_size);
		error_normalized = (float*)realloc(error_normalized, memory_size);
		neuron_backup = (float*)realloc(neuron_backup, memory_size);
		neuron_normalized = (float*)realloc(neuron_normalized, memory_size);

		this->batch_size = batch_size;
		this->time_step = time_step;
	}
}
void Batch_Normalization::Save(ofstream &file) {
	for (int j = 0; j < number_maps; j++) file << gamma[j] << endl;
	for (int j = 0; j < number_maps; j++) file << beta[j] << endl;
	for (int j = 0; j < time_step * number_maps; j++) file << mean[j] << endl;
	for (int j = 0; j < time_step * number_maps; j++) file << variance[j] << endl;
}
void Batch_Normalization::Set_Optimizer(Optimizer *optimizer) {
	delete gamma_optimizer;
	gamma_optimizer = optimizer->Copy(number_maps);

	delete beta_optimizer;
	beta_optimizer = optimizer->Copy(number_maps);
}

double Batch_Normalization::Calculate_Gradient(double learning_rate) {
	double sum_gradient = 0, *gradient = new double[number_maps];

#pragma omp parallel for
	for (int j = 0; j < number_maps; j++) {
		double sum = 0, g;

		float *error_backup = &this->error_backup[j * map_size];
		float *neuron_normalized = &this->neuron_normalized[j * map_size];

		for (int h = 0; h < batch_size * time_step; h++) {
			int index = h * number_nodes;

			for (int k = 0; k < map_size; k++) {
				sum += error_backup[index + k] * neuron_normalized[index + k];
			}
		}
		g = gamma_optimizer->Calculate_Gradient(j, sum, learning_rate);
		gamma_optimizer->gradient[j] = sum;
		gradient[j] = g * g;

		sum = 0;
		for (int h = 0; h < batch_size * time_step; h++) {
			int index = h * number_nodes;

			for (int k = 0; k < map_size; k++) {
				sum += error_backup[index + k];
			}
		}
		g = beta_optimizer->Calculate_Gradient(j, sum, learning_rate);
		beta_optimizer->gradient[j] = sum;
		gradient[j] += g * g;
	}
	for (int j = 0; j < number_maps; j++) {
		sum_gradient += gradient[j];
	}
	delete[] gradient;

	return sum_gradient;
}


Connection::Connection(string properties) {
	this->properties = properties;

	from_error = nullptr;
	from_neuron = nullptr;
	from_weight = nullptr;
	optimizer = nullptr;
	weight = nullptr;
	LSTM_weight = nullptr;
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
	if (weight) {
		delete[] weight;
	}
	if (LSTM_weight) {
		delete LSTM_weight;
	}
}

void Connection::Initialize(double scale) {
	if (weight) {
		for (int j = 0; j < number_weights; j++) {
			weight[j] = scale * (2.0 * rand() / RAND_MAX - 1);
		}
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
			for (int j = 0; j < number_weights; j++) {
				LSTM_weight->weight[h][j] = scale * (2.0 * rand() / RAND_MAX - 1);
			}
		}
	}
}
void Connection::Load(ifstream &file) {
	if (weight) {
		for (int j = 0; j < number_weights; j++) {
			file >> weight[j];
		}
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
			for (int j = 0; j < number_weights; j++) {
				file >> LSTM_weight->weight[h][j];
			}
		}
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
	if (weight) {
		for (int j = 0; j < number_weights; j++) {
			file << weight[j] << endl;
		}
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
			for (int j = 0; j < number_weights; j++) {
				file << LSTM_weight->weight[h][j] << endl;
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
		delete this->optimizer;
		this->optimizer = optimizer->Copy(number_weights);
	}
	if (LSTM_weight) {
		for (int h = 0; h < LSTM_weight->number_weight_types; h++) {
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

double Connectionist_Temporal_Classification::Backward_Algorithm(vector<string> label_sequence, int length_event, double **beta, float _likelihood[]) {
	double log_likelihood = 0;

	for (int t = length_event - 1, length_label_sequence = static_cast<int>(label_sequence.size()); t >= 0; t--) {
		float *likelihood = &_likelihood[t * number_labels];

		double sum = -numeric_limits<double>::infinity();

		if (t == length_event - 1) {
			for (int s = 0; s < length_label_sequence; s++) {
				beta[t][s] = log((s >= length_label_sequence - 2) * likelihood[Search_Label(label_sequence[s])]);
			}
		}
		else {
			for (int s = 0; s < length_label_sequence; s++) {
				double sum = -numeric_limits<double>::infinity();

				if (s <= 2 * t + 1) {
					if (label_sequence[s] == "" || (s <= length_label_sequence - 3 && label_sequence[s + 2] == label_sequence[s])) {
						sum = (s == length_label_sequence - 1) ? (beta[t + 1][s]) : (Log_Add(beta[t + 1][s], beta[t + 1][s + 1]));
					}
					else {
						sum = (s == length_label_sequence - 2) ? (Log_Add(beta[t + 1][s], beta[t + 1][s + 1])) : (Log_Add(Log_Add(beta[t + 1][s], beta[t + 1][s + 1]), beta[t + 1][s + 2]));
					}
				}
				beta[t][s] = sum + log(likelihood[Search_Label(label_sequence[s])]);
			}
		}
		for (int s = 0; s < length_label_sequence; s++) {
			sum = Log_Add(sum, beta[t][s]);
		}
		for (int s = 0; s < length_label_sequence; s++) {
			beta[t][s] -= sum;
		}
		log_likelihood += sum;
	}
	return log_likelihood;
}
double Connectionist_Temporal_Classification::Forward_Algorithm(vector<string> label_sequence, int length_event, double **alpha, float _likelihood[]) {
	double log_likelihood = 0;

	for (int t = 0, length_label_sequence = static_cast<int>(label_sequence.size()); t < length_event; t++) {
		float *likelihood = &_likelihood[t * number_labels];

		double sum = -numeric_limits<double>::infinity();

		if (t == 0) {
			for (int s = 0; s < length_label_sequence; s++) {
				alpha[t][s] = log((s <= 1) * likelihood[Search_Label(label_sequence[s])]);
			}
		}
		else {
			for (int s = 0; s < length_label_sequence; s++) {
				double sum = -numeric_limits<double>::infinity();

				if (s >= (length_label_sequence - 1) - 2 * ((length_event - 1) - t) - 1) {
					if (label_sequence[s] == "" || (s >= 2 && label_sequence[s - 2] == label_sequence[s])) {
						sum = (s == 0) ? (alpha[t - 1][s]) : (Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1]));
					}
					else {
						sum = (s == 1) ? (Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1])) : (Log_Add(Log_Add(alpha[t - 1][s], alpha[t - 1][s - 1]), alpha[t - 1][s - 2]));
					}
				}
				alpha[t][s] = sum + log(likelihood[Search_Label(label_sequence[s])]);
			}
		}
		for (int s = 0; s < length_label_sequence; s++) {
			sum = Log_Add(sum, alpha[t][s]);
		}
		for (int s = 0; s < length_label_sequence; s++) {
			alpha[t][s] -= sum;
		}
		log_likelihood += sum;
	}
	return log_likelihood;
}
double Connectionist_Temporal_Classification::Log_Add(double a, double b) {
	if (!isfinite(a) && !isfinite(b)) {
		return -numeric_limits<double>::infinity();
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

bool comparator(const pair<double, string> &a, const pair<double, string> &b) {
	return a.first > b.first;
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

double Connectionist_Temporal_Classification::Calculate_Error(vector<string> target_label_sequence, int length_event, float error[], float likelihood[]) {
	double log_likelihood;

	double **alpha = new double*[length_event];
	double **beta = new double*[length_event];

	for (int t = 0; t < length_event; t++) {
		alpha[t] = new double[target_label_sequence.size()];
		beta[t] = new double[target_label_sequence.size()];
	}

	log_likelihood = Forward_Algorithm(target_label_sequence, length_event, alpha, likelihood);
	Backward_Algorithm(target_label_sequence, length_event, beta, likelihood);

	for (int t = 0; t < length_event; t++) {
		int index = t * number_labels;

		for (int i = 0; i < number_labels; i++) {
			double sum[2] = { -numeric_limits<double>::infinity(), -numeric_limits<double>::infinity() };

			for (int j = 0; j < target_label_sequence.size(); j++) {
				int k = Search_Label(target_label_sequence[j]);

				if (i == k) {
					sum[1] = Log_Add(sum[1], alpha[t][j] + beta[t][j]);
				}
				sum[0] = Log_Add(sum[0], alpha[t][j] + beta[t][j] - log(likelihood[index + k]));
			}
			error[index + i] = likelihood[index + i] - exp(sum[1] - log(likelihood[index + i]) - sum[0]);

			if (!isfinite(error[index + i])) {
				error[index + i] = 0;
			}
		}
	}

	for (int t = 0; t < length_event; t++) {
		delete[] alpha[t];
		delete[] beta[t];
	}
	delete[] alpha;
	delete[] beta;

	return log_likelihood;
}


Layer::Layer(string properties, int number_maps, int map_width, int map_height, int map_depth) {
	this->map_width = map_width;
	this->map_height = map_height;
	this->map_depth = map_depth;
	this->number_maps = number_maps;
	this->properties = properties;

	Construct();
}
Layer::~Layer() {
	if (batch_normalization[0]) {
		delete batch_normalization[0];
	}
	if (batch_normalization[1]) {
		delete batch_normalization[1];
	}
	if (dropout_mask) {
		delete[] dropout_mask;
	}
	if (error[0]) {
		delete[] error[0];
	}
	if (error[1]) {
		delete[] error[1];
	}
	if (neuron[0]) {
		delete[] neuron[0];
	}
	if (neuron[1]) {
		delete[] neuron[1];
	}
	if (neuron[2]) {
		delete[] neuron[2];
	}
	if (bias) {
		delete[] bias;
	}
	if (bias_optimizer) {
		delete bias_optimizer;
	}
	if (slope_optimizer) {
		delete[] slope;
		delete slope_optimizer;
	}
	if (time_mask) {
		delete[] time_mask;
	}
	if (LSTM_node) {
		delete LSTM_node;
	}
}

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

	bias = nullptr;
	bias_optimizer = nullptr;
	dropout_mask = nullptr;
	slope = nullptr;
	slope_optimizer = nullptr;
	time_mask = nullptr;
	LSTM_node = nullptr;

	if (strstr(properties.c_str(), "dropout")) {
		dropout_mask = new bool[number_maps];
	}
	if (strstr(properties.c_str(), "LSTM")) {
		error[0] = new float[number_nodes];
		neuron[0] = new float[number_nodes];
		LSTM_node = new LSTM_Node(number_maps, map_size, batch_normalization);
	}
	else if (strstr(properties.c_str(), "RNN")) {
		if (batch_normalization) {
			this->batch_normalization[0] = new Batch_Normalization(number_maps, map_size);
			this->batch_normalization[1] = new Batch_Normalization(number_maps, map_size);
		}
		error[0] = new float[number_nodes];
		error[1] = new float[number_nodes];
		neuron[0] = new float[number_nodes];
		neuron[1] = new float[number_nodes];
	}
	else {
		if (batch_normalization) {
			this->batch_normalization[0] = new Batch_Normalization(number_maps, map_size);
		}
		if (strstr(properties.c_str(), "ELU")) {
			double parameter = atof(strstr(properties.c_str(), "ELU") + 3);

			slope = new float[1];
			*slope = parameter;
		}
		else if (strstr(properties.c_str(), "PReLU")) {
			double slope = atof(strstr(properties.c_str(), "PReLU") + 5);

			neuron[2] = new float[number_nodes];
			this->slope = new float[number_nodes];

			for (int j = 0; j < number_nodes; j++) {
				this->slope[j] = slope;
			}
			slope_optimizer = new Optimizer();
			slope_optimizer->Resize_Memory(number_nodes);
		}
		else if (strstr(properties.c_str(), "ReLU")) {
			double slope = atof(strstr(properties.c_str(), "ReLU") + 4);

			this->slope = new float[1];
			*this->slope = slope;
		}
		error[0] = new float[number_nodes];
		neuron[0] = new float[number_nodes];
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
		for (int j = 0; j < number_maps; j++) {
			bias[j] = scale * (2.0 * rand() / RAND_MAX - 1);
		}
	}
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Initialize(gamma);
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Initialize(gamma);
		}
		for (int h = 0; h < LSTM_node->number_weight_types; h++) {
			for (int j = 0; j < number_maps; j++) {
				LSTM_node->bias[h][j] = (h == LSTM_node->forget) ? (1) : (scale * (2.0 * rand() / RAND_MAX - 1));
			}
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			for (int j = 0; j < number_maps; j++) {
				LSTM_node->peephole[h][j] = scale * (2.0 * rand() / RAND_MAX - 1);
			}
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
	if (bias) {
		for (int j = 0; j < number_maps; j++) {
			file >> bias[j];
		}
	}
	if (slope_optimizer) {
		for (int j = 0; j < number_nodes; j++) {
			file >> slope[j];
		}
	}
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Load(file);
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Load(file);
		}
		for (int h = 0; h < LSTM_node->number_weight_types; h++) {
			for (int j = 0; j < number_maps; j++) {
				file >> LSTM_node->bias[h][j];
			}
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			for (int j = 0; j < number_maps; j++) {
				file >> LSTM_node->peephole[h][j];
			}
		}
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
			dropout_mask = (bool*)realloc(dropout_mask, batch_size * number_maps);
		}
		if (error[0]) {
			error[0] = (float*)realloc(error[0], memory_size);
		}
		if (error[1]) {
			error[1] = (float*)realloc(error[1], memory_size);
		}
		if (neuron[0]) {
			neuron[0] = (float*)realloc(neuron[0], memory_size);
		}
		if (neuron[1]) {
			neuron[1] = (float*)realloc(neuron[1], memory_size);
		}
		if (neuron[2]) {
			neuron[2] = (float*)realloc(neuron[2], memory_size);
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
	if (bias) {
		for (int j = 0; j < number_maps; j++) {
			file << bias[j] << endl;
		}
	}
	if (slope_optimizer) {
		for (int j = 0; j < number_nodes; j++) {
			file << slope[j] << endl;
		}
	}
	if (LSTM_node) {
		for (int h = 0; h < LSTM_node->number_node_types; h++) {
			if (LSTM_node->batch_normalization[h][0]) LSTM_node->batch_normalization[h][0]->Save(file);
			if (LSTM_node->batch_normalization[h][1]) LSTM_node->batch_normalization[h][1]->Save(file);
		}
		for (int h = 0; h < LSTM_node->number_weight_types; h++) {
			for (int j = 0; j < number_maps; j++) {
				file << LSTM_node->bias[h][j] << endl;
			}
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			for (int j = 0; j < number_maps; j++) {
				file << LSTM_node->peephole[h][j] << endl;
			}
		}
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
		delete bias_optimizer;
		bias_optimizer = optimizer->Copy(number_maps);
	}
	if (slope_optimizer) {
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
			delete LSTM_node->bias_optimizer[h];
			LSTM_node->bias_optimizer[h] = optimizer->Copy(number_maps);
		}
		for (int h = 0; h < LSTM_node->number_weight_types - 1; h++) {
			delete LSTM_node->peephole_optimizer[h];
			LSTM_node->peephole_optimizer[h] = optimizer->Copy(number_maps);
		}
	}
	delete optimizer;
}
void Layer::Set_Time_Mask(bool time_mask[]) {
	if (this->time_mask) {
		delete[] this->time_mask;
	}
	this->time_mask = time_mask;
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
			connection->weight = new float[connection->number_weights];

			if (bias == nullptr) {
				bias = new float[number_maps];
			}
			if (bias_optimizer == nullptr) {
				bias_optimizer = new Optimizer();
				bias_optimizer->Resize_Memory(number_maps);
			}
		}
	}

	if (properties[0] == 'P' || properties[0] == 'W') {
		bool depthwise_separable = (strstr(properties.c_str(), "DS") != 0);

		connection->from_error = new vector<Index>[parent_layer->number_nodes];
		connection->from_neuron = new vector<Index>[number_nodes];
		connection->from_weight = new vector<Index>[connection->number_weights];

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

															connection->from_error[node_index[1]].push_back(index);
															connection->from_neuron[node_index[0]].push_back(index);
															connection->from_weight[index.weight].push_back(index);
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

													connection->from_error[node_index[1]].push_back(index);
													connection->from_neuron[node_index[0]].push_back(index);
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

													connection->from_error[node_index[1]].push_back(index);
													connection->from_neuron[node_index[0]].push_back(index);
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
		bias[h] = new float[number_maps];
		bias_optimizer[h] = new Optimizer();
		bias_optimizer[h]->Resize_Memory(number_maps);
	}
	for (int h = 0; h < number_weight_types - 1; h++) {
		peephole[h] = new float[number_maps];
		peephole_optimizer[h] = new Optimizer();
		peephole_optimizer[h]->Resize_Memory(number_maps);
	}
	for (int h = 0; h < number_node_types; h++) {
		error[h][0] = new float[number_nodes];
		error[h][1] = new float[number_nodes];
		neuron[h][0] = new float[number_nodes];
		neuron[h][1] = new float[number_nodes];
	}
}
LSTM_Node::~LSTM_Node() {
	for (int h = 0; h < number_node_types; h++) {
		if (batch_normalization[h][0]) delete batch_normalization[h][0];
		if (batch_normalization[h][1]) delete batch_normalization[h][1];

		delete[] error[h][0];
		delete[] error[h][1];
		delete[] neuron[h][0];
		delete[] neuron[h][1];
	}
	for (int h = 0; h < number_weight_types; h++) {
		delete[] bias[h];
		delete bias_optimizer[h];
	}
	for (int h = 0; h < number_weight_types - 1; h++) {
		delete[] peephole[h];
		delete peephole_optimizer[h];
	}
}

void LSTM_Node::Resize_Memory(int batch_size, int time_step) {
	for (int h = 0; h < number_node_types; h++) {
		int memory_size = sizeof(float) * batch_size * time_step * number_nodes;

		if (batch_normalization[h][0]) batch_normalization[h][0]->Resize_Memory(batch_size, time_step);
		if (batch_normalization[h][1]) batch_normalization[h][1]->Resize_Memory(batch_size, time_step);

		error[h][0] = (float*)realloc(error[h][0], memory_size);
		error[h][1] = (float*)realloc(error[h][1], memory_size);
		neuron[h][0] = (float*)realloc(neuron[h][0], memory_size);
		neuron[h][1] = (float*)realloc(neuron[h][1], memory_size);
	}
}


LSTM_Weight::LSTM_Weight(int number_weights) {
	for (int h = 0; h < number_weight_types; h++) {
		this->optimizer[h] = new Optimizer();
		this->optimizer[h]->Resize_Memory(number_weights);
		weight[h] = new float[number_weights];
	}
}
LSTM_Weight::~LSTM_Weight() {
	for (int h = 0; h < number_weight_types; h++) {
		delete optimizer[h];
		delete[] weight[h];
	}
}


void Neural_Networks::Activate(Layer *layer, string phase, int time_index) {
	int t = time_index;

	if (layer->batch_normalization[0]) layer->batch_normalization[0]->Activate(phase, layer->neuron[0], time_index);
	if (layer->batch_normalization[1]) layer->batch_normalization[1]->Activate(phase, layer->neuron[1], time_index);

#pragma omp parallel for
	for (int j = 0; j < layer->number_maps; j++) {
		for (int h = 0; h < batch_size; h++) {
			int index = (h * time_step + t) * layer->number_nodes + j * layer->map_size;

			for (int k = 0; k < layer->map_size; k++) {
				if (layer->neuron[1]) {
					layer->neuron[0][index + k] += layer->neuron[1][index + k];
				}
				if (layer->bias) {
					layer->neuron[0][index + k] += layer->bias[j];
				}
			}
		}
	}

	if (strstr(layer->properties.c_str(), "LSTM")) {
		bool backward = (strstr(layer->properties.c_str(), "backward") != 0);

		LSTM_Node *LSTM_node = layer->LSTM_node;

		float *input_bias = LSTM_node->bias[LSTM_node->input];
		float *forget_bias = LSTM_node->bias[LSTM_node->forget];
		float *output_bias = LSTM_node->bias[LSTM_node->output];
		float *cell_bias = LSTM_node->bias[LSTM_node->cell];

		float *input_neuron[] = { LSTM_node->neuron[LSTM_node->input][0], LSTM_node->neuron[LSTM_node->input][1] };
		float *forget_neuron[] = { LSTM_node->neuron[LSTM_node->forget][0], LSTM_node->neuron[LSTM_node->forget][1] };
		float *output_neuron[] = { LSTM_node->neuron[LSTM_node->output][0], LSTM_node->neuron[LSTM_node->output][1] };
		float *cell_neuron[] = { LSTM_node->neuron[LSTM_node->cell][0], LSTM_node->neuron[LSTM_node->cell][1] };
		float *cell_output_neuron = LSTM_node->neuron[LSTM_node->cell_output][0];

		for (int i = 0; i < LSTM_node->number_node_types - 1; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Activate(phase, LSTM_node->neuron[i][0], time_index);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Activate(phase, LSTM_node->neuron[i][1], time_index);
		}

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			float *previous_cell_output_neuron = (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (LSTM_node->neuron[LSTM_node->cell_output][0]);

			for (int h = 0, k = j / layer->map_size; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				input_neuron[0][index] += input_neuron[1][index] + input_bias[k];
				forget_neuron[0][index] += forget_neuron[1][index] + forget_bias[k];
				output_neuron[0][index] += output_neuron[1][index] + output_bias[k];
				cell_neuron[0][index] += cell_neuron[1][index] + cell_bias[k];

				if (backward == false && t) {
					input_neuron[0][index] += LSTM_node->peephole[LSTM_node->input][k] * previous_cell_output_neuron[index - layer->number_nodes];
					forget_neuron[0][index] += LSTM_node->peephole[LSTM_node->forget][k] * previous_cell_output_neuron[index - layer->number_nodes];
					output_neuron[0][index] += LSTM_node->peephole[LSTM_node->output][k] * previous_cell_output_neuron[index - layer->number_nodes];
				}
				else if (backward == true && t + 1 < time_step) {
					input_neuron[0][index] += LSTM_node->peephole[LSTM_node->input][k] * previous_cell_output_neuron[index + layer->number_nodes];
					forget_neuron[0][index] += LSTM_node->peephole[LSTM_node->forget][k] * previous_cell_output_neuron[index + layer->number_nodes];
					output_neuron[0][index] += LSTM_node->peephole[LSTM_node->output][k] * previous_cell_output_neuron[index + layer->number_nodes];
				}
				input_neuron[0][index] = 1 / (1 + exp(-input_neuron[0][index]));
				forget_neuron[0][index] = 1 / (1 + exp(-forget_neuron[0][index]));
				output_neuron[0][index] = 1 / (1 + exp(-output_neuron[0][index]));
				cell_neuron[0][index] = 2 / (1 + exp(-2 * cell_neuron[0][index])) - 1;

				if (backward == false && t) {
					cell_output_neuron[index] = forget_neuron[0][index] * previous_cell_output_neuron[index - layer->number_nodes];
				}
				else if (backward == true && t + 1 < time_step) {
					cell_output_neuron[index] = forget_neuron[0][index] * previous_cell_output_neuron[index + layer->number_nodes];
				}
				else {
					cell_output_neuron[index] = 0;
				}
				cell_output_neuron[index] += input_neuron[0][index] * cell_neuron[0][index];
			}
		}

		if (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) {
			LSTM_node->batch_normalization[LSTM_node->cell_output][0]->Activate(phase, cell_output_neuron, time_index);
		}

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				layer->neuron[0][index] = output_neuron[0][index] * (2 / (1 + exp(-2 * cell_output_neuron[index])) - 1);
			}
		}
	}
	else if (strstr(layer->properties.c_str(), "RNN")) {
#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				layer->neuron[0][index] = 2 / (1 + exp(-2 * layer->neuron[0][index])) - 1;
			}
		}
	}
	else {
		if (strstr(layer->properties.c_str(), "ELU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->neuron[0][index] = (layer->neuron[0][index] > 0) ? (layer->neuron[0][index]) : (*layer->slope * (exp(layer->neuron[0][index]) - 1));
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "PReLU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->neuron[2][index] = layer->neuron[0][index];
					layer->neuron[0][index] *= (layer->neuron[2][index] > 0) ? (1) : (layer->slope[j]);
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "ReLU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->neuron[0][index] *= (layer->neuron[0][index] > 0) ? (1) : (*layer->slope);
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "sigmoid")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->neuron[0][index] = 1 / (1 + exp(-layer->neuron[0][index]));
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "softmax")) {
#pragma omp parallel for
			for (int h = 0; h < batch_size; h++) {
				double max;
				double sum = 0;

				float *neuron = &layer->neuron[0][(h * time_step + t) * layer->number_nodes];

				for (int j = 0; j < layer->number_nodes; j++) {
					if (j == 0 || max < neuron[j]) {
						max = neuron[j];
					}
				}
				for (int j = 0; j < layer->number_nodes; j++) {
					neuron[j] = exp(neuron[j] - max);
					sum += neuron[j];
				}
				for (int j = 0; j < layer->number_nodes; j++) {
					neuron[j] /= sum;
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "tangent")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->neuron[0][index] = 2 / (1 + exp(-2 * layer->neuron[0][index])) - 1;
				}
			}
		}

		if (strstr(layer->properties.c_str(), "dropout")) {
			double rate = atof(strstr(layer->properties.c_str(), "dropout") + 7);

			if (phase == "training") {
#pragma omp parallel for
				for (int j = 0; j < layer->number_maps; j++) {
					for (int h = 0; h < batch_size; h++) {
						int index = (h * time_step + t) * layer->number_nodes + j * layer->map_size;

						for (int k = 0; k < layer->map_size; k++) {
							if (layer->dropout_mask[h * layer->number_maps + j] == false) {
								layer->neuron[0][index + k] = 0;
							}
						}
					}
				}
			}
			else if (phase == "inference") {
#pragma omp parallel for
				for (int j = 0; j < layer->number_nodes; j++) {
					for (int h = 0; h < batch_size; h++) {
						int index = (h * time_step + t) * layer->number_nodes + j;

						layer->neuron[0][index] *= rate;
					}
				}
			}
		}
	}
}
void Neural_Networks::Adjust_Parameter(Layer *layer, double gradient_clip, double learning_rate) {
	for (int g = 0; g < layer->parent_layer.size(); g++) {
		if (layer->connection[g]->properties[0] == 'W') {
			Connection *connection = layer->connection[g];

			if (connection->LSTM_weight) {
				LSTM_Weight *LSTM_weight = connection->LSTM_weight;

#pragma omp parallel for
				for (int j = 0; j < connection->number_weights; j++) {
					for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
						LSTM_weight->weight[i][j] += LSTM_weight->optimizer[i]->Calculate_Gradient(j, LSTM_weight->optimizer[i]->gradient[j], gradient_clip * learning_rate, true);
					}
				}
			}
			else {
#pragma omp parallel for
				for (int j = 0; j < connection->number_weights; j++) {
					connection->weight[j] += connection->optimizer->Calculate_Gradient(j, connection->optimizer->gradient[j], gradient_clip * learning_rate, true);
				}
			}
		}
	}

	if (layer->LSTM_node) {
		LSTM_Node *LSTM_node = layer->LSTM_node;

#pragma omp parallel for
		for (int j = 0; j < layer->number_maps; j++) {
			for (int i = 0; i < LSTM_node->number_weight_types; i++) {
				LSTM_node->bias[i][j] += LSTM_node->bias_optimizer[i]->Calculate_Gradient(j, LSTM_node->bias_optimizer[i]->gradient[j], gradient_clip * learning_rate, true);
			}
			for (int i = 0; i < LSTM_node->number_weight_types - 1; i++) {
				LSTM_node->peephole[i][j] += LSTM_node->peephole_optimizer[i]->Calculate_Gradient(j, LSTM_node->peephole_optimizer[i]->gradient[j], gradient_clip * learning_rate, true);
			}
		}
		for (int i = 0; i < LSTM_node->number_node_types; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Adjust_Parameter(gradient_clip, learning_rate);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Adjust_Parameter(gradient_clip, learning_rate);
		}
	}
	if (layer->bias_optimizer) {
#pragma omp parallel for
		for (int j = 0; j < layer->number_maps; j++) {
			layer->bias[j] += layer->bias_optimizer->Calculate_Gradient(j, layer->bias_optimizer->gradient[j], gradient_clip * learning_rate, true);
		}
	}
	if (layer->slope_optimizer) {
#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			layer->slope[j] += layer->slope_optimizer->Calculate_Gradient(j, layer->slope_optimizer->gradient[j], gradient_clip * learning_rate, true);
		}
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
#pragma omp parallel for
			for (int j = 0; j < parent_layer->number_nodes; j++) {
				vector<Index> &from_error = connection->from_error[j];

				if (connection->properties[0] == 'P') {
					for (int h = 0; h < batch_size; h++) {
						float sum = 0, *error = &layer->error[0][(h * time_step + t) * layer->number_nodes];

						for (auto index = from_error.begin(); index != from_error.end(); index++) {
							sum += error[(index)->next_node];
						}
						parent_layer->error[0][(h * time_step + t) * parent_layer->number_nodes + j] += sum;
					}
				}
				else if (connection->properties[0] == 'W') {
					if (recurrent == false) {
						if (layer->LSTM_node) {
							LSTM_Node *LSTM_node = layer->LSTM_node;

							LSTM_Weight *LSTM_weight = connection->LSTM_weight;

							for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
								for (int h = 0; h < batch_size; h++) {
									float sum = 0, *error = &LSTM_node->error[i][0][(h * time_step + t) * layer->number_nodes];

									for (auto index = from_error.begin(); index != from_error.end(); index++) {
										sum += error[(index)->next_node] * LSTM_weight->weight[i][(index)->weight];
									}
									parent_layer->error[0][(h * time_step + t) * parent_layer->number_nodes + j] += sum;
								}
							}
						}
						else {
							for (int h = 0; h < batch_size; h++) {
								float sum = 0, *error = &layer->error[0][(h * time_step + t) * layer->number_nodes];

								for (auto index = from_error.begin(); index != from_error.end(); index++) {
									sum += error[(index)->next_node] * connection->weight[(index)->weight];
								}
								parent_layer->error[0][(h * time_step + t) * parent_layer->number_nodes + j] += sum;
							}
						}
					}
					else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
						if (layer->LSTM_node) {
							LSTM_Node *LSTM_node = layer->LSTM_node;

							LSTM_Weight *LSTM_weight = connection->LSTM_weight;

							for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
								for (int h = 0; h < batch_size; h++) {
									float sum = 0, *error = &LSTM_node->error[i][1][(h * time_step + t) * layer->number_nodes];

									for (auto index = from_error.begin(); index != from_error.end(); index++) {
										sum += error[(index)->next_node] * LSTM_weight->weight[i][(index)->weight];
									}
									parent_layer->error[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes + j] += sum;
								}
							}
						}
						else {
							for (int h = 0; h < batch_size; h++) {
								float sum = 0, *error = &layer->error[1][(h * time_step + t) * layer->number_nodes];

								for (auto index = from_error.begin(); index != from_error.end(); index++) {
									sum += error[(index)->next_node] * connection->weight[(index)->weight];
								}
								parent_layer->error[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes + j] += sum;
							}
						}
					}
				}
				else if (strstr(connection->properties.c_str(), "add")) {
					for (int h = 0; h < batch_size; h++) {
						parent_layer->error[0][(h * time_step + t) * parent_layer->number_nodes + j] += layer->error[0][(h * time_step + t) * layer->number_nodes + j];
					}
				}
				else if (strstr(connection->properties.c_str(), "dilate")) {
					for (int h = 0; h < batch_size; h++) {
						parent_layer->error[0][(h * time_step + t) * parent_layer->number_nodes + j] += layer->error[0][(h * time_step + t) * layer->number_nodes + (from_error.begin())->next_node];
					}
				}
			}
		}
	}
}
void Neural_Networks::Feedforward(Layer *layer, int time_index, bool backward) {
	int t = time_index;

#pragma omp parallel for
	for (int j = 0; j < layer->number_nodes; j++) {
		for (int g = 0; g < layer->parent_layer.size(); g++) {
			bool recurrent = (strstr(layer->connection[g]->properties.c_str(), "recurrent") != 0);

			Connection *connection = layer->connection[g];

			Layer *parent_layer = layer->parent_layer[g];

			vector<Index> &from_neuron = connection->from_neuron[j];

			if (recurrent || parent_layer->Check_Mask(t)) {
				if (connection->properties[0] == 'P') {
					if (strstr(connection->properties.c_str(), "average")) {
						for (int h = 0; h < batch_size; h++) {
							float sum = 0, *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

							for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
								sum += neuron[(index)->prev_node];
							}
							layer->neuron[0][(h * time_step + t) * layer->number_nodes + j] = sum / from_neuron.size();
						}
					}
					else if (strstr(connection->properties.c_str(), "max")) {
						for (int h = 0; h < batch_size; h++) {
							float max, *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

							for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
								if (index == from_neuron.begin() || max < neuron[(index)->prev_node]) {
									max = neuron[(index)->prev_node];
								}
							}
							layer->neuron[0][(h * time_step + t) * layer->number_nodes + j] = max;
						}
					}
				}
				else if (connection->properties[0] == 'W') {
					if (recurrent == false) {
						if (layer->LSTM_node) {
							LSTM_Node *LSTM_node = layer->LSTM_node;

							LSTM_Weight *LSTM_weight = connection->LSTM_weight;

							for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
								for (int h = 0; h < batch_size; h++) {
									float sum = 0, *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

									for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
										sum += neuron[(index)->prev_node] * LSTM_weight->weight[i][(index)->weight];
									}
									LSTM_node->neuron[i][0][(h * time_step + t) * layer->number_nodes + j] += sum;
								}
							}
						}
						else {
							for (int h = 0; h < batch_size; h++) {
								float sum = 0, *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

								for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
									sum += neuron[(index)->prev_node] * connection->weight[(index)->weight];
								}
								layer->neuron[0][(h * time_step + t) * layer->number_nodes + j] += sum;
							}
						}
					}
					else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
						if (layer->LSTM_node) {
							LSTM_Node *LSTM_node = layer->LSTM_node;

							LSTM_Weight *LSTM_weight = connection->LSTM_weight;

							for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
								for (int h = 0; h < batch_size; h++) {
									float sum = 0, *neuron = &parent_layer->neuron[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes];

									for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
										sum += neuron[(index)->prev_node] * LSTM_weight->weight[i][(index)->weight];
									}
									LSTM_node->neuron[i][1][(h * time_step + t) * layer->number_nodes + j] += sum;
								}
							}
						}
						else {
							for (int h = 0; h < batch_size; h++) {
								float sum = 0, *neuron = &parent_layer->neuron[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes];

								for (auto index = from_neuron.begin(); index != from_neuron.end(); index++) {
									sum += neuron[(index)->prev_node] * connection->weight[(index)->weight];
								}
								layer->neuron[1][(h * time_step + t) * layer->number_nodes + j] += sum;
							}
						}
					}
				}
				else if (strstr(connection->properties.c_str(), "add")) {
					for (int h = 0; h < batch_size; h++) {
						layer->neuron[0][(h * time_step + t) * layer->number_nodes + j] += parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes + j];
					}
				}
				else if (strstr(connection->properties.c_str(), "dilate")) {
					for (int h = 0; h < batch_size; h++) {
						layer->neuron[0][(h * time_step + t) * layer->number_nodes + j] = parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes + (from_neuron.begin())->prev_node];
					}
				}
			}
		}
	}
}

void Neural_Networks::FloatToNode(float **memory, vector<Layer*> &layer) {
	for (int i = 0; i < layer.size(); i++) {
		memcpy(layer[i]->neuron[0], memory[i], sizeof(float) * batch_size * time_step * layer[i]->number_nodes);
	}
}
void Neural_Networks::FloatToNode(float ***memory, vector<Layer*> &layer, int length_data[]) {
	for (int i = 0; i < layer.size(); i++) {
		for (int h = 0; h < batch_size; h++) {
			memset(&layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], 0, sizeof(float) * time_step * layer[i]->number_nodes);
			memcpy(&layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], memory[h][i], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[h])) * layer[i]->number_nodes);
		}
	}
}
void Neural_Networks::NodeToFloat(vector<Layer*> &layer, float ***memory) {
	for (int i = 0; i < layer.size(); i++) {
		for (int h = 0; h < batch_size; h++) {
			memcpy(memory[h][i], &layer[i]->neuron[0][h * time_step * layer[i]->number_nodes], sizeof(float) * time_step * layer[i]->number_nodes);
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
				memset(layer->neuron[0], 0, memory_size);
			}
			if (layer->neuron[1]) {
				memset(layer->neuron[1], 0, memory_size);
			}
			if (layer->LSTM_node) {
				LSTM_Node *LSTM_node = layer->LSTM_node;

				for (int i = 0; i < LSTM_node->number_weight_types; i++) {
					memset(LSTM_node->neuron[i][0], 0, memory_size);
					memset(LSTM_node->neuron[i][1], 0, memory_size);
				}
			}
			memset(layer->error[0], 0, memory_size);
		}
	}
}

double Neural_Networks::Calculate_Gradient(Layer *layer, double learning_rate, bool backward) {
	double sum_gradient = 0;

	for (int g = 0; g < layer->parent_layer.size(); g++) {
		if (layer->connection[g]->properties[0] == 'W') {
			bool recurrent = (strstr(layer->connection[g]->properties.c_str(), "recurrent") != 0);

			double *gradient = new double[layer->connection[g]->number_weights];

			Connection *connection = layer->connection[g];

			Layer *parent_layer = layer->parent_layer[g];

			memset(gradient, 0, sizeof(double) * connection->number_weights);

			if (connection->LSTM_weight) {
				LSTM_Node *LSTM_node = layer->LSTM_node;

				LSTM_Weight *LSTM_weight = connection->LSTM_weight;

#pragma omp parallel for
				for (int j = 0; j < connection->number_weights; j++) {
					for (int i = 0; i < LSTM_weight->number_weight_types; i++) {
						double sum = 0, g;

						vector<Index> &from_weight = connection->from_weight[j];

						for (int t = 0; t < time_step; t++) {
							if (recurrent == false) {
								if (parent_layer->Check_Mask(t)) {
									for (int h = 0; h < batch_size; h++) {
										float *error = &LSTM_node->error[i][0][(h * time_step + t) * layer->number_nodes];
										float *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

										for (auto index = from_weight.begin(); index != from_weight.end(); index++) {
											sum += error[(index)->next_node] * neuron[(index)->prev_node];
										}
									}
								}
							}
							else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
								for (int h = 0; h < batch_size; h++) {
									float *error = &LSTM_node->error[i][1][(h * time_step + t) * layer->number_nodes];
									float *neuron = &parent_layer->neuron[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes];

									for (auto index = from_weight.begin(); index != from_weight.end(); index++) {
										sum += error[(index)->next_node] * neuron[(index)->prev_node];
									}
								}
							}
						}
						g = LSTM_weight->optimizer[i]->Calculate_Gradient(j, sum, learning_rate);
						LSTM_weight->optimizer[i]->gradient[j] = sum;
						gradient[j] += g * g;
					}
				}
			}
			else {
#pragma omp parallel for
				for (int j = 0; j < connection->number_weights; j++) {
					double sum = 0, g;

					vector<Index> &from_weight = connection->from_weight[j];

					for (int t = 0; t < time_step; t++) {
						if (recurrent == false) {
							if (parent_layer->Check_Mask(t)) {
								for (int h = 0; h < batch_size; h++) {
									float *error = &layer->error[0][(h * time_step + t) * layer->number_nodes];
									float *neuron = &parent_layer->neuron[0][(h * time_step + t) * parent_layer->number_nodes];

									for (auto index = from_weight.begin(); index != from_weight.end(); index++) {
										sum += error[(index)->next_node] * neuron[(index)->prev_node];
									}
								}
							}
						}
						else if ((backward == false && t) || (backward == true && t + 1 < time_step)) {
							for (int h = 0; h < batch_size; h++) {
								float *error = &layer->error[1][(h * time_step + t) * layer->number_nodes];
								float *neuron = &parent_layer->neuron[0][((backward == false) ? (h * time_step + t - 1) : (h * time_step + t + 1)) * parent_layer->number_nodes];

								for (auto index = from_weight.begin(); index != from_weight.end(); index++) {
									sum += error[(index)->next_node] * neuron[(index)->prev_node];
								}
							}
						}
					}
					g = connection->optimizer->Calculate_Gradient(j, sum, learning_rate);
					connection->optimizer->gradient[j] = sum;
					gradient[j] += g * g;
				}
			}
			for (int j = 0; j < connection->number_weights; j++) {
				sum_gradient += gradient[j];
			}
			delete[] gradient;
		}
	}

	if (layer->LSTM_node) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);

		double *gradient = new double[layer->number_maps];

		LSTM_Node *LSTM_node = layer->LSTM_node;

		memset(gradient, 0, sizeof(double) * layer->number_maps);

#pragma omp parallel for
		for (int j = 0; j < layer->number_maps; j++) {
			float *previous_cell_output_neuron = (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (LSTM_node->neuron[LSTM_node->cell_output][0]);

			for (int i = 0; i < LSTM_node->number_weight_types; i++) {
				double sum = 0, g;

				for (int h = 0; h < batch_size * time_step; h++) {
					int index = h * layer->number_nodes + j * layer->map_size;

					for (int k = 0; k < layer->map_size; k++) {
						sum += (batch_normalization) ? (LSTM_node->batch_normalization[i][0]->error_backup[index + k]) : (LSTM_node->error[i][0][index + k]);
					}
				}
				g = LSTM_node->bias_optimizer[i]->Calculate_Gradient(j, sum, learning_rate);
				LSTM_node->bias_optimizer[i]->gradient[j] = sum;
				gradient[j] += g * g;
			}
			for (int i = 0; i < LSTM_node->number_weight_types - 1; i++) {
				double sum = 0, g;

				for (int h = 0; h < batch_size * time_step; h++) {
					int index = h * layer->number_nodes + j * layer->map_size;

					if (backward == false && h % time_step) {
						for (int k = 0; k < layer->map_size; k++) {
							sum += ((batch_normalization) ? (LSTM_node->batch_normalization[i][0]->error_backup[index + k]) : (LSTM_node->error[i][0][index + k])) * previous_cell_output_neuron[index + k - layer->number_nodes];
						}
					}
					else if (backward == true && (h % time_step) + 1 < time_step) {
						for (int k = 0; k < layer->map_size; k++) {
							sum += ((batch_normalization) ? (LSTM_node->batch_normalization[i][0]->error_backup[index + k]) : (LSTM_node->error[i][0][index + k])) * previous_cell_output_neuron[index + k + layer->number_nodes];
						}
					}
				}
				g = LSTM_node->peephole_optimizer[i]->Calculate_Gradient(j, sum, learning_rate);
				LSTM_node->peephole_optimizer[i]->gradient[j] = sum;
				gradient[j] += g * g;
			}
		}
		for (int j = 0; j < layer->number_maps; j++) {
			sum_gradient += gradient[j];
		}
		delete[] gradient;

		for (int i = 0; i < LSTM_node->number_node_types; i++) {
			if (LSTM_node->batch_normalization[i][0]) sum_gradient += LSTM_node->batch_normalization[i][0]->Calculate_Gradient(learning_rate);
			if (LSTM_node->batch_normalization[i][1]) sum_gradient += LSTM_node->batch_normalization[i][1]->Calculate_Gradient(learning_rate);
		}
	}
	if (layer->bias_optimizer) {
		bool batch_normalization = (strstr(layer->properties.c_str(), "BN") != 0);

		double *gradient = new double[layer->number_maps];

#pragma omp parallel for
		for (int j = 0; j < layer->number_maps; j++) {
			double sum = 0, g;

			for (int h = 0; h < batch_size * time_step; h++) {
				int index = h * layer->number_nodes + j * layer->map_size;

				for (int k = 0; k < layer->map_size; k++) {
					sum += (batch_normalization) ? (layer->batch_normalization[0]->error_backup[index + k]) : (layer->error[0][index + k]);
				}
			}
			g = layer->bias_optimizer->Calculate_Gradient(j, sum, learning_rate);
			layer->bias_optimizer->gradient[j] = sum;
			gradient[j] = g * g;
		}
		for (int j = 0; j < layer->number_maps; j++) {
			sum_gradient += gradient[j];
		}
		delete[] gradient;
	}
	if (layer->slope_optimizer) {
		double *gradient = new double[layer->number_nodes];

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			double sum = 0, g;

			for (int h = 0; h < batch_size * time_step; h++) {
				int index = h * layer->number_nodes + j;

				sum += (layer->neuron[2][index] > 0) ? (0) : (layer->error[0][index] * layer->neuron[2][index]);
			}
			g = layer->slope_optimizer->Calculate_Gradient(j, sum, learning_rate);
			layer->slope_optimizer->gradient[j] = sum;
			gradient[j] = g * g;
		}
		for (int j = 0; j < layer->number_nodes; j++) {
			sum_gradient += gradient[j];
		}
		delete[] gradient;
	}
	if (layer->batch_normalization[0]) {
		sum_gradient += layer->batch_normalization[0]->Calculate_Gradient(learning_rate);
	}
	if (layer->batch_normalization[1]) {
		sum_gradient += layer->batch_normalization[1]->Calculate_Gradient(learning_rate);
	}
	return sum_gradient;
}
double Neural_Networks::Differentiate(Layer *layer, float target_output[], int time_index) {
	int t = time_index;

	double sum = 0;

	if (target_output && layer->Check_Mask(t) && (strstr(layer->properties.c_str(), "CE") || strstr(layer->properties.c_str(), "MSE"))) {
		int loss_type = (strstr(layer->properties.c_str(), "CE")) ? (0) : (1);

		double *loss = new double[layer->number_nodes];

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			double sum = 0;

			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				// calculate error
				layer->error[0][index] = layer->neuron[0][index] - target_output[index];

				// calculate loss
				if (loss_type == 0) {
					sum -= target_output[index] * log(layer->neuron[0][index] + 0.000001) + (1 - target_output[index]) * log(1 - layer->neuron[0][index] + 0.000001);
				}
				else {
					sum += 0.5 * (layer->neuron[0][index] - target_output[index]) * (layer->neuron[0][index] - target_output[index]);
				}
			}
			loss[j] = sum;
		}
		for (int j = 0; j < layer->number_nodes; j++) {
			sum += loss[j];
		}
		delete[] loss;
	}

	if (strstr(layer->properties.c_str(), "LSTM")) {
		bool backward = (strstr(layer->properties.c_str(), "backward") != 0);

		LSTM_Node *LSTM_node = layer->LSTM_node;

		float *input_error[] = { LSTM_node->error[LSTM_node->input][0], LSTM_node->error[LSTM_node->input][1] };
		float *forget_error[] = { LSTM_node->error[LSTM_node->forget][0], LSTM_node->error[LSTM_node->forget][1] };
		float *output_error[] = { LSTM_node->error[LSTM_node->output][0], LSTM_node->error[LSTM_node->output][1] };
		float *cell_error[] = { LSTM_node->error[LSTM_node->cell][0], LSTM_node->error[LSTM_node->cell][1] };
		float *cell_output_error = LSTM_node->error[LSTM_node->cell_output][0];

		float *input_neuron = LSTM_node->neuron[LSTM_node->input][0];
		float *forget_neuron = LSTM_node->neuron[LSTM_node->forget][0];
		float *output_neuron = LSTM_node->neuron[LSTM_node->output][0];
		float *cell_neuron = LSTM_node->neuron[LSTM_node->cell][0];
		float *cell_output_neuron = LSTM_node->neuron[LSTM_node->cell_output][0];

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				float active_cell_output_neuron = 2 / (1 + exp(-2 * cell_output_neuron[index])) - 1;

				output_error[0][index] = layer->error[0][index] * active_cell_output_neuron;
				cell_output_error[index] = layer->error[0][index] * output_neuron[index] * (1 - active_cell_output_neuron) * (1 + active_cell_output_neuron);
			}
		}

		if (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) {
			LSTM_node->batch_normalization[LSTM_node->cell_output][0]->Differentiate(cell_output_error, time_index);
		}

#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			float *next_input_error = (LSTM_node->batch_normalization[LSTM_node->input][0]) ? (LSTM_node->batch_normalization[LSTM_node->input][0]->error_backup) : (input_error[0]);
			float *next_forget_error = (LSTM_node->batch_normalization[LSTM_node->forget][0]) ? (LSTM_node->batch_normalization[LSTM_node->forget][0]->error_backup) : (forget_error[0]);
			float *next_output_error = (LSTM_node->batch_normalization[LSTM_node->output][0]) ? (LSTM_node->batch_normalization[LSTM_node->output][0]->error_backup) : (output_error[0]);
			float *previous_cell_output_neuron = (LSTM_node->batch_normalization[LSTM_node->cell_output][0]) ? (LSTM_node->batch_normalization[LSTM_node->cell_output][0]->neuron_backup) : (cell_output_neuron);

			for (int h = 0, k = j / layer->map_size; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				if (backward == false && t + 1 < time_step) {
					cell_output_error[index] += next_input_error[index + layer->number_nodes] * LSTM_node->peephole[LSTM_node->input][k];
					cell_output_error[index] += next_forget_error[index + layer->number_nodes] * LSTM_node->peephole[LSTM_node->forget][k];
					cell_output_error[index] += next_output_error[index + layer->number_nodes] * LSTM_node->peephole[LSTM_node->output][k];
					cell_output_error[index] += cell_output_error[index + layer->number_nodes] * forget_neuron[index + layer->number_nodes];
				}
				else if (backward == true && t) {
					cell_output_error[index] += next_input_error[index - layer->number_nodes] * LSTM_node->peephole[LSTM_node->input][k];
					cell_output_error[index] += next_forget_error[index - layer->number_nodes] * LSTM_node->peephole[LSTM_node->forget][k];
					cell_output_error[index] += next_output_error[index - layer->number_nodes] * LSTM_node->peephole[LSTM_node->output][k];
					cell_output_error[index] += cell_output_error[index - layer->number_nodes] * forget_neuron[index - layer->number_nodes];
				}
				input_error[0][index] = cell_output_error[index] * cell_neuron[index];

				if (backward == false && t) {
					forget_error[0][index] = cell_output_error[index] * previous_cell_output_neuron[index - layer->number_nodes];
				}
				else if (backward == true && t + 1 < time_step) {
					forget_error[0][index] = cell_output_error[index] * previous_cell_output_neuron[index + layer->number_nodes];
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

		for (int i = 0; i < LSTM_node->number_node_types - 1; i++) {
			if (LSTM_node->batch_normalization[i][0]) LSTM_node->batch_normalization[i][0]->Differentiate(LSTM_node->error[i][0], time_index);
			if (LSTM_node->batch_normalization[i][1]) LSTM_node->batch_normalization[i][1]->Differentiate(LSTM_node->error[i][1], time_index);
		}
	}
	else if (strstr(layer->properties.c_str(), "RNN")) {
#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				layer->error[0][index] *= (1 - layer->neuron[0][index]) * (1 + layer->neuron[0][index]);
			}
		}
	}
	else {
		if (strstr(layer->properties.c_str(), "ELU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->error[0][index] *= (layer->neuron[0][index] > 0) ? (1) : (layer->neuron[0][index] + *layer->slope);
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "PReLU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->error[0][index] *= (layer->neuron[2][index] > 0) ? (1) : (layer->slope[j]);
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "ReLU")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->error[0][index] *= (layer->neuron[0][index] > 0) ? (1) : (*layer->slope);
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "sigmoid") && !strstr(layer->properties.c_str(), "CE")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->error[0][index] *= (1 - layer->neuron[0][index]) * layer->neuron[0][index];
				}
			}
		}
		else if (strstr(layer->properties.c_str(), "softmax")) {
			// error = error;
		}
		else if (strstr(layer->properties.c_str(), "tangent")) {
#pragma omp parallel for
			for (int j = 0; j < layer->number_nodes; j++) {
				for (int h = 0; h < batch_size; h++) {
					int index = (h * time_step + t) * layer->number_nodes + j;

					layer->error[0][index] *= (1 - layer->neuron[0][index]) * (1 + layer->neuron[0][index]);
				}
			}
		}
	}

	if (layer->error[1]) {
#pragma omp parallel for
		for (int j = 0; j < layer->number_nodes; j++) {
			for (int h = 0; h < batch_size; h++) {
				int index = (h * time_step + t) * layer->number_nodes + j;

				layer->error[1][index] = layer->error[0][index];
			}
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

#pragma omp parallel for
		for (int h = 0; h < batch_size; h++) {
			vector<string> label_sequence;

			for (int j = 0; j < target_label_sequence[h].size(); j++) {
				label_sequence.push_back("");
				label_sequence.push_back(target_label_sequence[h][j]);
			}
			label_sequence.push_back("");

			log_likelihood[h] = CTC->Calculate_Error(label_sequence, (length_data) ? (length_data[h]) : (time_step), &layer->error[0][h * time_step * layer->number_nodes], &layer->neuron[0][h * time_step * layer->number_nodes]);
		}
		for (int h = 0; h < batch_size; h++) {
			sum += log_likelihood[h];
		}
		delete[] log_likelihood;
	}
	for (int t = time_step - 1; t >= 0; t--) {
		Differentiate(layer, nullptr, t);
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

	double sum = 0;

	float **input_batch = new float*[layer[0].size()];
	float **target_output_batch = (target_output) ? (new float*[layer[layer_height - 1].size()]) : (nullptr);

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
		input_batch[j] = new float[batch_size * time_step * layer[i][j]->number_nodes];
	}
	for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
		target_output_batch[j] = new float[batch_size * time_step * layer[i][j]->number_nodes];
	}
	Resize_Memory(batch_size);
	Set_Epsilon(epsilon);

	for (int g = 0, h = 0; g < number_training; g++) {
		for (int i = 0, j = 0; j < layer[i].size(); j++) {
			memset(&input_batch[j][h * time_step * layer[i][j]->number_nodes], 0, sizeof(float) * time_step * layer[i][j]->number_nodes);
			memcpy(&input_batch[j][h * time_step * layer[i][j]->number_nodes], input[index[g]][j], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[index[g]])) * layer[i][j]->number_nodes);
		}
		for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
			memset(&target_output_batch[j][h * time_step * layer[i][j]->number_nodes], 0, sizeof(float) * time_step * layer[i][j]->number_nodes);
			memcpy(&target_output_batch[j][h * time_step * layer[i][j]->number_nodes], target_output[index[g]][j], sizeof(float) * ((length_data == nullptr) ? (time_step) : (length_data[index[g]])) * layer[i][j]->number_nodes);
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
				default_random_engine generator(rand());
				normal_distribution<double> distribution(0, noise_standard_deviation);

				for (int i = 0, j = 0; j < layer[i].size(); j++) {
					for (int k = 0; k < batch_size * time_step * layer[i][j]->number_nodes; k++) {
						input_batch[j][k] += distribution(generator);
					}
				}
			}

			for (int i = 0; i < layer_height; i++) {
				for (int j = 0; j < layer[i].size(); j++) {
					Layer *layer = this->layer[i][j];

					// initialize dropout mask
					if (strstr(layer->properties.c_str(), "dropout")) {
						double rate = atof(strstr(layer->properties.c_str(), "dropout") + 7);

						for (int k = 0; k < batch_size * layer->number_maps; k++) {
							layer->dropout_mask[k] = ((double)rand() / RAND_MAX <= rate);
						}
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
		delete[] input_batch[j];
	}
	for (int i = layer_height - 1, j = 0; j < layer[i].size() && target_output; j++) {
		delete[] target_output_batch[j];
	}
	if (target_output) {
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
	if (y >= layer_height || layer[y].size() <= x) {
		return nullptr;
	}
	return layer[y][x];
}


void Optimizer::Initialize(string type, double epsilon, double factor_1, double factor_2) {
	string name[] = { "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam" };

	this->epsilon = epsilon;
	this->factor_1 = factor_1;
	this->factor_2 = factor_2;
	this->type = 0;

	this->gradient = nullptr;
	this->momentum = nullptr;
	this->velocity = nullptr;

	for (int i = 0; i < 6; i++) {
		if (type == name[i]) {
			this->type = i + 1;
			break;
		}
	}
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
Optimizer::~Optimizer() {
	if (gradient) delete[] gradient;
	if (momentum) delete[] momentum;
	if (velocity) delete[] velocity;
}

void Optimizer::Resize_Memory(int number_parameters) {
	if (number_parameters == 0) {
		return;
	}
	gradient = (gradient == nullptr) ? (new float[number_parameters]) : ((float*)realloc(gradient, sizeof(float) * number_parameters));

	switch (type) {
	case 1:
	case 2:
	case 3:
	case 4:
		memset(velocity = (velocity == nullptr) ? (new float[number_parameters]) : ((float*)realloc(velocity, sizeof(float) * number_parameters)), 0, sizeof(float) * number_parameters);
		break;
	case 5:
	case 6:
		memset(momentum = (momentum == nullptr) ? (new float[number_parameters]) : ((float*)realloc(momentum, sizeof(float) * number_parameters)), 0, sizeof(float) * number_parameters);
		memset(velocity = (velocity == nullptr) ? (new float[number_parameters]) : ((float*)realloc(velocity, sizeof(float) * number_parameters)), 0, sizeof(float) * number_parameters);
		break;
	}
}

float Optimizer::Calculate_Gradient(int parameter_index, double _gradient, double learning_rate, bool update) {
	float &gradient = this->gradient[parameter_index];

	if (type == 0) {
		return (gradient = -learning_rate * _gradient);
	}

	float &velocity = this->velocity[parameter_index];

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

	float &momentum = this->momentum[parameter_index];

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

Optimizer* Optimizer::Copy(int number_parameters) {
	string name[] = { "", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam" };

	Optimizer *optimizer = new Optimizer(name[type], epsilon, factor_1, factor_2);

	optimizer->Resize_Memory(number_parameters);

	return optimizer;
}