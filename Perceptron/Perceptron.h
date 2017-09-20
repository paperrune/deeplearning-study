class Perceptron{
private:
	int number_layer;

	int *number_neuron;

	double **neuron;
	double **derivative;

	double ***weight;

	void Activate(int layer_index, int neuron_index);
	void Adjust_Parameter(int layer_index, int neuron_index);
	void Differentiate(int layer_index, int neuron_index, double learning_rate, double target_output[]);
	void Feedforward(int layer_index, int neuron_index);
public:
	Perceptron(int number_neuron[]);
	~Perceptron();

	void Initialize_Parameter(int seed, double scale, double shift);
	void Test(double input[], double output[]);

	double Train(int number_training, double learning_rate, double **input, double **target_output);
};