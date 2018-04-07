#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class LSTM_Node;
class LSTM_Weight;
class Optimizer;

typedef struct Index {
	int next_node;
	int prev_node;
	int weight;
} Index;

class Batch_Normalization {
public:
	int batch_size;
	int map_size;
	int number_maps;
	int number_nodes;
	int time_step;

	float *beta;
	float *gamma;
	float *mean;
	float *sum_mean;
	float *sum_variance;
	float *variance;

	float *error_backup;
	float *error_normalized;
	float *neuron_backup;
	float *neuron_normalized;

	double epsilon;

	Optimizer *gamma_optimizer;
	Optimizer *beta_optimizer;

	Batch_Normalization(int number_maps, int map_size);
	~Batch_Normalization();

	void Activate(string phase, float neuron[], int time_index);
	void Adjust_Parameter(double learning_rate);
	void Calculate_Mean_Variance(int number_batches);
	void Destroy();
	void Differentiate(float error[], int time_index);
	void Initialize(double gamma);
	void Load(ifstream &file);
	void Resize_Memory(int batch_size, int time_step);
	void Save(ofstream &file);
	void Set_Optimizer(Optimizer *optimizer);

	double Calculate_Gradient(double learning_rate);
};

class Connection {
public:
	int kernel_depth;
	int kernel_height;
	int kernel_size;
	int kernel_width;
	int number_maps;
	int number_nodes;
	int number_weights;
	int stride_depth;
	int stride_height;
	int stride_width;

	float *weight;

	string properties;

	vector<Index> *from_error;
	vector<Index> *from_neuron;
	vector<Index> *from_weight;

	Index *from_errors;
	Index *from_neurons;
	Index *from_weights;

	LSTM_Weight *LSTM_weight;

	Optimizer *optimizer;

	Connection(string properties);
	~Connection();

	void Destroy();
	void Initialize(double scale);
	void Load(ifstream &file);
	void Load(string path);
	void Save(ofstream &file);
	void Save(string path);
	void Set_Optimizer(Optimizer *optimier);
};

class Connectionist_Temporal_Classification {
private:
	unordered_map<string, int> label_index;

	int Search_Label(string label);

	double Backward_Algorithm(vector<string> label_sequence, int length_event, double **beta, float likelihood[]);
	double Forward_Algorithm(vector<string> label_sequence, int length_event, double **alpha, float likelihood[]);
public:
	int number_labels;

	string *label;

	Connectionist_Temporal_Classification(int number_labels, string label[]);
	~Connectionist_Temporal_Classification();

	void Best_Path_Decoding(int length_event, float likelihood[], vector<string> &label_sequence);
	void Prefix_Search_Decoding(int length_event, float likelihood[], vector<string> &label_sequence, int k);

	double Calculate_Error(vector<string> target_label_sequence, int length_event, float error[], float likelihood[]);
};

class Layer {
public:
	bool *dropout_mask;
	bool *time_mask;
	bool *time_mask_device;

	int batch_size;
	int index[2];
	int map_depth;
	int map_height;
	int map_size;
	int map_width;
	int number_connections;
	int number_maps;
	int number_nodes;
	int time_step;

	float *bias;
	float *error[2];
	float *slope;
	float *neuron[3];

	string properties;

	vector<Connection*> connection;

	vector<Layer*> parent_layer;

	Batch_Normalization *batch_normalization[2];

	LSTM_Node *LSTM_node;

	Optimizer *bias_optimizer;
	Optimizer *slope_optimizer;

	Layer(string properties, int number_maps, int map_width = 1, int map_height = 1, int map_depth = 1);
	~Layer();

	void Construct();
	void Destroy();
	void Disconnect(Layer *target_layer);
	void Initialize(double scale, double gamma = 1);
	void Load(ifstream &file);
	void Load(string path);
	void Resize_Memory(int batch_size, int time_step);
	void Save(ofstream &file);
	void Save(string path);
	void Set_Epsilon(double epsilon);
	void Set_Optimizer(Optimizer *optimizer);
	void Set_Time_Mask(bool time_mask[]);

	bool Check_Mask(int time_index);

	Connection* Connect(Layer *parent_layer, string properties);
};

class LSTM_Node {
public:
	enum LSTM { input = 0, forget = 1, output = 2, cell = 3, cell_output = 4, number_node_types = 5, number_weight_types = 4 };

	int number_nodes;

	float *bias[number_weight_types];
	float *error[number_node_types][2];
	float *neuron[number_node_types][2];
	float *peephole[number_weight_types - 1];

	Batch_Normalization *batch_normalization[number_node_types][2];

	Optimizer *bias_optimizer[number_weight_types];
	Optimizer *peephole_optimizer[number_weight_types - 1];

	LSTM_Node(int number_maps, int map_size, bool batch_normalization);
	~LSTM_Node();

	void Destroy();
	void Resize_Memory(int batch_size, int time_step);
};

class LSTM_Weight {
public:
	enum LSTM { input = 0, forget = 1, output = 2, cell = 3, cell_output = 4, number_node_types = 5, number_weight_types = 4 };

	float *weight[number_weight_types];

	Optimizer *optimizer[number_weight_types];

	LSTM_Weight(int number_weight);
	~LSTM_Weight();

	void Destroy();
};

class Neural_Networks {
private:
	int batch_size;
	int time_step;

	double epsilon;
	double gradient_threshold;

	Connectionist_Temporal_Classification *CTC;

	void Activate(Layer *layer, string phase, int time_index);
	void Adjust_Parameter(Layer *layer, double gradient_clip);
	void Backpropagate(Layer *layer, int time_index, bool reverse = false);
	void Feedforward(Layer *layer, int time_index, bool reverse = false);

	void FloatToNode(float **memory, vector<Layer*> &layer);
	void FloatToNode(float ***memory, vector<Layer*> &layer, int **length_data = nullptr);
	void NodeToFloat(vector<Layer*> &layer, float ***memory);
	void Resize_Memory(int batch_size, int time_step = 0);

	double Calculate_Gradient(Layer *layer, double learning_rate, bool reverse = false);
	double Differentiate(Layer *layer, float target_output[], int time_index);
	double Differentiate(Layer *layer, vector<string> target_label_sequence[]);
public:
	int layer_depth;

	vector<vector<Layer*>> layer;

	Neural_Networks(string path);
	Neural_Networks(int time_step = 1);
	~Neural_Networks();

	void Decode(int length_event, float likelihood[], vector<string> &label_sequence, int k = 0);
	void Initialize(double scale, double gamma = 1);
	void Save(string path);
	void Set_CTC_Loss(int number_labels, string label[]);
	void Set_Epsilon(double epsilon);
	void Set_Gradient_Threshold(double gradient_threshold);
	void Set_Optimizer(Optimizer *optimizer);
	void Test(float input[], float output[], int length_data = 0);
	void Test(int batch_size, float **input, float **output, int length_data[] = nullptr);
	void Test(int batch_size, float ***input, float ***output, int **length_data = nullptr);

	double Train(int batch_size, int number_training, float **input, float **target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0);
	double Train(int batch_size, int number_training, int length_data[], float **input, float **target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0);
	double Train(int batch_size, int number_training, float **input, vector<string> target_label_sequence[], double learning_rate, double epsilon = 0, double noise_standard_deviation = 0);
	double Train(int batch_size, int number_training, int length_data[], float **input, vector<string> target_label_sequence[], double learning_rate, double epsilon = 0, double noise_standard_deviation = 0);
	double Train(int batch_size, int number_training, int length_data[], float ***input, float ***target_output, vector<string> target_label_sequence[], double learning_rate, double epsilon = 0, double noise_standard_deviation = 0);

	Layer* Add(Layer *layer, int index = -1);
	Layer* Get_Layer(int x, int y);
};

class Optimizer {
public:
	int type = 0;

	float *gradient;
	float *momentum;
	float *velocity;

	double factor_1;
	double factor_2;
	double epsilon;

	void Initialize(string type, double epsilon, double factor_1, double factor_2);

	Optimizer();
	Optimizer(string type, double momentum_epsilon);
	Optimizer(string type, double decay_rate, double epsilon);
	Optimizer(string type, double epsilon, double beta_1, double beta_2);
	~Optimizer();

	void Resize_Memory(int number_parameters);

	double Calculate_Gradient(int parameter_index, double gradient, double learning_rate);

	Optimizer* Copy(int number_parameters = 0);
	Optimizer* Copy_Host(int number_parameters = 0);
};

#endif