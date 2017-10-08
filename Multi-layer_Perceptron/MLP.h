class Multilayer_Perceptron{
private:
	char **type_layer;

	int batch_size;
	int number_layers;
	int number_memory_types;

	int *number_neurons;

	double ***weight;

	double ****neuron;
	double ****derivative;

	// Variables for Batch Normalization
	double epsilon;

	double **gamma;
	double **beta;
	double **mean;
	double **variance;
	double **sum_mean;
	double **sum_variance;
	// *********************************

	void Activate(char option[], int layer_index, int neuron_index);
	void Adjust_Parameter(int layer_index, int neuron_index);
	void Backpropagate(int layer_index, int neuron_index);
	void Differentiate(int layer_index, int neuron_index, double learning_rate, double **target_output);
	void Feedforward(int layer_index, int neuron_index);
	void Softmax(int layer_index);

	void Batch_Normalization_Activate(char option[], int layer_index, int neuron_index);
	void Batch_Normalization_Adjust_Parameter(int layer_index, int neuron_index);
	void Batch_Normalization_Differentiate(int layer_index, int neuron_index);

	void Resize_Memory(int batch_size);

	bool Access_Memory(int type_index, int layer_index);
public:
	Multilayer_Perceptron(char **type_layer, int number_layers, int number_neurons[]);
	~Multilayer_Perceptron();

	void Initialize_Parameter(int seed, double scale, double shift);
	void Test(double input[], double output[]);

	double Train(int batch_size, int number_training, double epsilon, double learning_rate, double **input, double **target_output);
};
