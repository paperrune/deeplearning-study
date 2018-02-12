#ifndef Neual_Netweorks_H
#define Neual_Netweorks_H

#include <string>
#include <vector>

using namespace std;

enum LSTM { input = 0, forget = 1, output = 2, cell = 3, cell_output = 4, number_node_types = 5, number_weight_types = 3 };

typedef struct connection {
	int lower_node;
	int upper_node;
	int weight;
} connection;

class Batch_Normalization {
private:
	int batch_size;
	int time_step;
	int map_size;

	double epsilon;

	double *derivative_normalized;
	double *neuron_normalized;
	double *sum_mean;
	double *sum_variance;
public:
	double gamma;
	double beta;

	double *mean;
	double *variance;

	double *derivative_backup;
	double *neuron_backup;

	Batch_Normalization();
	Batch_Normalization(int batch_size, int map_size, int time_step);
	~Batch_Normalization();

	void Activate(string option, int number_nodes, int time_index, double neuron[]);
	void Adjust_Parameter();
	void Calculate_Mean_Variance(int number_batches = 0);
	void Construct(int batch_size, int map_size, int time_step);
	void Differentiate(int number_nodes, int time_index, double derivative[]);
	void Resize_Node(int batch_size, double epsilon);
};

class Neural_Networks{
private:
	char **type_layer;

	int batch_size = 1;
	int number_batch_types = 2;
	int number_node_types = 3;
	int number_layers;
	int time_step;

	int *kernel_width;
	int *kernel_height;
	int *map_size;
	int *map_width;
	int *map_height;
	int *number_maps;
	int *number_nodes;
	int *number_weights;
	int *stride_width;
	int *stride_height;

	double epsilon;

	double **derivative[3];
	double **neuron[3];
	double **recurrent_weight;
	double **weight;

	double **lstm_derivative[LSTM::number_node_types][3];
	double **lstm_neuron[LSTM::number_node_types][3];
	double **lstm_recurrent_weight[LSTM::number_weight_types];
	double **lstm_weight[LSTM::number_weight_types];

	Batch_Normalization **batch_normalization[2];
	Batch_Normalization **lstm_batch_normalization[LSTM::number_node_types][2];

	vector<connection> **from_derivative[3];
	vector<connection> **from_neuron[3];

	void Activate(char option[], int layer_index, int map_index, int time_index);
	void Adjust_Parameter(int layer_index, int map_index);
	void Backpropagate(int layer_index, int map_index);
	void Differentiate(int layer_index, int map_index, int time_index, bool output_maks[], double learning_rate, double target_output[]);
	void Feedforward(int layer_index, int map_index);
	void LSTM_Backward(int layer_index, int map_index, int time_index);
	void LSTM_Forward(string option, int layer_index, int map_index, int time_index);
	void Recurrent_Backward(int layer_index, int map_index, int time_index);
	void Recurrent_Forward(int layer_index, int map_index, int time_index);
	void Softmax(int layer_index);

	void Construct_Networks();
	void Resize_Node(int batch_size);

	bool Access_Node(int type_index, int layer_index);
public:
	Neural_Networks(string path);
	Neural_Networks(string type_layer[], int number_layers, int number_maps[], int map_width[] = nullptr, int map_height[] = nullptr, int time_step = 1);
	~Neural_Networks();

	void Initialize_Parameter(double mean, double variance, double gamma = 1, int seed = 0);
	void Save_Model(string path);
	void Test(double input[], double output[]);
	void Test(int batch_size, double **input, double **output);

	double Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output, bool output_mask[] = nullptr, double noise_variance = 0);
};

#endif