class Recurrent_Neural_Networks{
private:
	char **type_layer;

	int batch_size;
	int number_layers;
	int number_memory_batches;
	int number_memory_parts;
	int number_memory_types;
	int number_weight_types;
	int networks_time_step;
	int time_step;
	int test_time_index;

	int *kernel_width;
	int *kernel_height;
	int *map_width;
	int *map_height;
	int *number_maps;
	int *stride_width;
	int *stride_height;

	int ***dropout_mask;

	double gradient_factor;

	double ***cell_weight;
	double ***cell_weight_momentum;

	double ******weight;
	double ******weight_momentum;

	double ********derivative;
	double ********neuron;

	// Variables for Batch Normalization
	double epsilon;

	double ****gamma;
	double ****gamma_momentum;
	double ****beta;
	double ****beta_momentum;

	double *****mean;
	double *****variance;
	double *****sum_mean;
	double *****sum_variance;
	// *********************************

	void Activate(char option[], int layer_index, int time_index, int map_index);
	void Adjust_Parameter(int layer_index, int time_index, int map_index);
	void Adjust_Parameter(int layer_index, int map_index, double ****lower_neuron, double ****previous_neuron, double ****cell_neuron, double ****reset_neuron, double ****_derivative, double *****derivative_patch, double *cell_weight, double ****weight);
	void Backpropagate(char option, int layer_index, int time_index, int map_index);
	void Backpropagate(bool initialize, int layer_index, int map_index, double ****derivative, double ****upper_derivative, double *****upper_weight);
	void Differentiate(int layer_index, int time_index, int map_index, bool output_mask[], double learning_rate, double ***target_output);
	void Feedforward(char option[], int layer_index, int time_index, int map_index);
	void Feedforward(int layer_index, int map_index, double ****neuron, double *****neuron_patch, double ****lower_neuron, double ****previous_neuron, double ****cell_neuron, double ****reset_neuron, double *cell_weight, double ****weight);
	void Softmax(int layer_index, int time_index);

	void Batch_Normalization_Activate(char option[], int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index);
	void Batch_Normalization_Adjust_Parameter(int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index);
	void Batch_Normalization_Differentiate(int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index);

	void Gradient_Clipping(double threshold);
	void Refer_Memory(char option[], int time_index);
	void Refer_Parameter(char option[], char type_parameter_A[], char type_parameter_B[], double factor);
	void Resize_Memory(int batch_size, int time_step);

	bool Access_Memory(int memory_type, int memory_patch_index, int layer_index);
	bool Access_Weight(int weight_type, int layer_index);

	double Tangent(double x);
	double Sigmoid(double x);
public:
	Recurrent_Neural_Networks(char **type_layer, int number_layers, int time_step, int map_width[], int map_height[], int number_maps[]);
	~Recurrent_Neural_Networks();

	void Initialize_Parameter(int seed, double scale, double shift);
	void Load_Parameter(char path[]);
	void Save_Parameter(char path[]);
	void Test(bool initialize, int time_index, double input[], double output[]);

	double Train(int batch_size, int number_training, int time_step, int length_data[], bool output_mask[], double epsilon, double gradient_threshold, double learning_rate, double noise_scale_factor, double ***input, double ***target_output);
};
