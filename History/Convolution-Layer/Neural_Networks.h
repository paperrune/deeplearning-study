#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Index;
struct Layer;
struct Optimizer;

namespace Activation {
	enum {linear, relu, sigmoid, softmax, tanh};
}
namespace Loss {
	enum { cross_entropy, mean_squared_error };
}

struct Connection {
	int kernel_width;
	int kernel_height;
	int kernel_depth;
	int kernel_size;
	int number_weights;
	int stride_width;
	int stride_height;
	int stride_depth;

	float *weight;

	string properties;

	Layer *layer;
	Layer *parent_layer;

	Optimizer *optimizer;

	vector<Index> *from_error;
	vector<Index> *from_neuron;
	vector<Index> *from_weight;

	Connection(Layer *layer, Layer *parent_layer, string properties);
	~Connection();

	Connection* Initialize(double scale);
};

struct Index {
	int next_node;
	int prev_node;
	int weight;
};

struct Layer {
	bool *mask;

	int activation;
	int batch_size;
	int map_width;
	int map_height;
	int map_depth;
	int map_size;
	int number_maps;
	int number_nodes;

	float *bias;
	float *error;
	float *neuron;	

	string properties;

	vector<Connection*> connection;
	vector<Connection*> child_connection;

	Optimizer *optimizer;

	Layer(int number_maps, int map_width, int map_height, int map_depth, string properties = "");
	~Layer();

	void Activate(bool training = false);
	void Backward();
	void Differentiate(int loss = -1, float **y_batch = nullptr);
	void Forward();
	void Resize_Memory(int batch_size);

	Layer* Activation(int activation);
	Layer* Initialize(double scale);
};

struct Optimizer {
	struct SGD {
		double decay_rate;
		double learning_rate;

		SGD(double learning_rate, double decay_rate = 0) {
			this->decay_rate = decay_rate;
			this->learning_rate = learning_rate;
		}
	};
	struct Momentum {
		double decay_rate;
		double learning_rate;
		double momentum;

		Momentum(double learning_rate, double momentum, double decay_rate = 0) {
			this->decay_rate = decay_rate;
			this->learning_rate = learning_rate;
			this->momentum = momentum;
		}
	};
	struct Nesterov {
		double decay_rate;
		double learning_rate;
		double momentum;

		Nesterov(double learning_rate, double momentum, double decay_rate = 0) {
			this->decay_rate = decay_rate;
			this->learning_rate = learning_rate;
			this->momentum = momentum;
		}
	};

	int type;

	float *gradient;

	double decay_rate;
	double learning_rate;
	double momentum;

	void Construct(int type, double learning_rate, double momentum, double decay_rate, int number_parameters);

	Optimizer(int type, double learning_rate, double momentum, double decay_rate, int number_parameters);
	Optimizer(SGD SGD);
	Optimizer(Momentum Momentum);
	Optimizer(Nesterov Nesterov);
	~Optimizer();

	double Calculate_Gradient(int index, double gradient, int iterations = 0);

	Optimizer* Copy(int number_parameters);
};

class Neural_Networks {
private:
	int batch_size;
	int iterations;
	int loss;

	vector<Connection*> connection;
	vector<Layer*> layer;

	Optimizer *optimizer;

	void Resize_Memory(int batch_size);

	double Calculate_Loss(Layer *layer, float **y_batch);
public:
	Neural_Networks();
	~Neural_Networks();

	void Compile(int loss, Optimizer *optimizer);
	void Predict(float input[], float output[]);
	void Predict(float **input, float **output, int batch_size = 1);

	float** Shuffle(int seed, float **data, int data_size);

	double Evaluate(float **x_data, float **y_data, int data_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);

	Connection* Connect(int from, int to, string properties);

	Layer* Add(int number_nodes, string properties = "");
	Layer* Add(int number_maps, int map_width, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, int map_depth, string properties = "");
};

#endif