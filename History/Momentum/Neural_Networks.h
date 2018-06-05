#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Layer;
struct Optimizer;

namespace Activation {
	enum {linear, relu, sigmoid, softmax, tanh};
}
namespace Loss {
	enum { cross_entropy, mean_squared_error };
}

struct Connection {
	int number_weights;

	float *weight;

	string activation;

	Layer *layer;
	Layer *parent_layer;

	Optimizer *optimizer;

	Connection(Layer *layer, Layer *parent_layer, double scale);
	~Connection();
};
struct Layer {
	bool *mask;

	int activation;
	int batch_size;
	int number_nodes;

	float *bias;
	float *error;
	float *neuron;	

	string properties;

	vector<Connection*> connection;

	Optimizer *optimizer;

	Layer(int number_nodes, int activation = 0, string properties = "");
	~Layer();

	void Activation(bool training = false);
	void Backward();
	void Derivative(int loss = -1, float **y_batch = nullptr);
	void Forward();
	void Resize_Memory(int batch_size);
};

struct Optimizer {
	enum { SGD };

	int type;

	float *gradient;

	double learning_rate;
	double momentum;

	Optimizer(int type, double learning_rate, double momentum = 0, int number_parameters = 0);
	~Optimizer();

	double Calculate_Gradient(int index, double gradient);

	Optimizer* Copy(int number_parameters);
};

class Neural_Networks {
private:
	int batch_size;
	int loss;

	vector<Connection*> connection;
	vector<Layer*> layer;

	Optimizer *optimizer;

	void Resize_Memory(int batch_size);

	double Calculate_Loss(Layer *layer, float **y_batch);
public:
	Neural_Networks();
	~Neural_Networks();

	void Add(int number_nodes, int activation = 0, string properties = "");
	void Compile(int loss, Optimizer *optimizer);
	void Connect(int from, int to, double scale);
	void Predict(float input[], float output[]);
	void Predict(float **input, float **output, int batch_size = 1);

	float** Shuffle(int seed, float **data, int data_size);

	double Evaluate(float **x_data, float **y_data, int data_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);
};

#endif