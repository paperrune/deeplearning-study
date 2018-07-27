#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Layer;

struct Connection {
	int number_weights;

	float *weight;

	Layer *layer;
	Layer *parent_layer;

	Connection(Layer *layer, Layer *parent_layer, double scale);
	~Connection();
};

struct Layer {
	int batch_size;
	int number_nodes;

	float *bias;
	float *error;
	float *neuron;

	vector<Connection*> connection;

	Layer(int number_nodes);
	~Layer();

	void Forward();
	void Resize_Memory(int batch_size);
};

class Neural_Networks {
private:
	int batch_size;

	double learning_rate;

	vector<Connection*> connection;
	vector<Layer*> layer;

	void Resize_Memory(int batch_size);

	double Calculate_Loss(Layer *layer, float **y_batch);
public:
	Neural_Networks();
	~Neural_Networks();

	void Add(int number_nodes);
	void Compile(double learning_rate);
	void Connect(int from, int to, double scale);
	void Predict(float input[], float output[]);
	void Predict(float **input, float **output, int batch_size = 1);

	double Evaluate(float **x_data, float **y_data, int test_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);
};

#endif
