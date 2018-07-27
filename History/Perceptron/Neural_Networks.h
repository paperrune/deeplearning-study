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

	void Destruct();
};

struct Layer {
	int number_nodes;

	float *bias;
	float *error;
	float *neuron;

	vector<Connection*> connection;

	Layer(int number_nodes);
	~Layer();

	void Destruct();
	void Forward();
};

class Neural_Networks {
private:
	double learning_rate;

	vector<Connection*> connection;
	vector<Layer*> layer;
public:
	Neural_Networks();
	~Neural_Networks();

	void Add(int number_nodes);
	void Compile(double learning_rate);
	void Connect(int from, int to, double scale);
	void Predict(float input[], float output[]);

	double Calculate_Loss(Layer *layer, float y_data[]);
	double Evaluate(float **x_data, float **y_data, int data_size);
	double Fit(float **x_train, float **y_train, int train_size);
};

#endif