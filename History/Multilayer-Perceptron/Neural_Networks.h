#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Layer;

namespace Activation {
	enum {linear, sigmoid, softmax};
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

	Connection(Layer *layer, Layer *parent_layer, double scale);
	~Connection();
};
struct Layer {
	int activation;
	int batch_size;
	int number_nodes;

	float *bias;
	float *error;
	float *neuron;	

	vector<Connection*> connection;

	Layer(int number_nodes, int activation = 0);
	~Layer();

	void Activation();
	void Backward();
	void Derivative(int loss = -1, float **y_batch = nullptr);
	void Forward();
	void Resize_Memory(int batch_size);
};

class Neural_Networks {
private:
	int batch_size;
	int loss;

	double learning_rate;

	vector<Connection*> connection;
	vector<Layer*> layer;

	void Resize_Memory(int batch_size);

	double Calculate_Loss(Layer *layer, float **y_batch);
public:
	Neural_Networks();
	~Neural_Networks();

	void Add(int number_nodes, int activation = 0);
	void Compile(int loss, double learning_rate);
	void Connect(int from, int to, double scale);
	void Predict(float input[], float output[]);
	void Predict(float **input, float **output, int batch_size = 1);

	double Evaluate(float **x_data, float **y_data, int data_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);
};

#endif