#ifndef CNN_H
#define CNN_H

#include <string>
#include <vector>

using namespace std;

typedef struct connection {
	int lower_node;
	int upper_node;
	int weight;
} connection;

class Convolutional_Neural_Networks{
private:
	char **type_layer;

	int batch_size;
	int number_layers;
	int number_node_types;

	int *kernel_width;
	int *kernel_height;
	int *map_area;
	int *map_width;
	int *map_height;
	int *number_maps;
	int *number_nodes;
	int *number_weights;
	int *stride_width;
	int *stride_height;

	double **weight;

	double ****neuron;
	double ****derivative;

	vector<connection> **connected_derivative;
	vector<connection> **connected_neuron;

	// Variables for Batch Normalization
	double epsilon;

	double **gamma;
	double **beta;
	double **mean;
	double **variance;
	double **sum_mean;
	double **sum_variance;
	// *********************************

	void Activate(char option[], int layer_index, int map_index);
	void Adjust_Parameter(int layer_index, int map_index);
	void Backpropagate(int layer_index, int map_index);
	void Differentiate(int layer_index, int map_index, double learning_rate, double **target_output);
	void Feedforward(int layer_index, int map_index);
	void Softmax(int layer_index);

	void Batch_Normalization_Activate(char option[], int layer_index, int map_index);
	void Batch_Normalization_Adjust_Parameter(int layer_index, int map_index);
	void Batch_Normalization_Differentiate(int layer_index, int map_index);

	void Construct_Networks();
	void Resize_Memory(int batch_size);

	bool Access_Memory(int type_index, int layer_index);
public:
	Convolutional_Neural_Networks(string path);
	Convolutional_Neural_Networks(string type_layer[], int number_layers, int number_maps[], int map_width[] = nullptr, int map_height[] = nullptr);
	~Convolutional_Neural_Networks();

	void Initialize_Parameter(double scale, double shift, int seed = 0);
	void Save_Model(string path);
	void Test(double input[], double output[]);

	double Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output);
};

#endif