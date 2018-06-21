#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Index;
struct Initializer;
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

	Connection* Initialize(Initializer initializer);
};

struct Index {
	int next_node;
	int prev_node;
	int weight;
};

struct Initializer {
	struct GlorotNormal {
		default_random_engine *generator;

		GlorotNormal(int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct GlorotUniform {
		default_random_engine *generator;

		GlorotUniform(int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct HeNormal {
		default_random_engine *generator;

		HeNormal(int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct HeUniform {
		default_random_engine *generator;

		HeUniform(int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct RandomNormal {
		double mean;
		double stdv;

		default_random_engine *generator;

		RandomNormal(double stdv, double mean = 0, int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
			this->mean = mean;
			this->stdv = stdv;
		}
	};
	struct RandomUniform {
		double max;
		double min;

		default_random_engine *generator;

		RandomUniform(double min, double max, int seed = -1) {
			generator = (seed >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
			this->max = max;
			this->min = min;
		}
	};

	int type;

	double max;
	double min;
	double mean;
	double stdv;
	double value;

	default_random_engine *generator;

	Initializer(double value);
	Initializer(GlorotNormal initializer);
	Initializer(GlorotUniform initializer);
	Initializer(HeNormal initializer);
	Initializer(HeUniform initializer);
	Initializer(RandomNormal initializer);
	Initializer(RandomUniform initializer);
	~Initializer();

	void Random(int memory_size, float memory[], int fan_in, int fan_out);
};

typedef Initializer::GlorotNormal GlorotNormal;
typedef Initializer::GlorotUniform GlorotUniform;
typedef Initializer::HeNormal HeNormal;
typedef Initializer::HeUniform HeUniform;
typedef Initializer::RandomNormal RandomNormal;
typedef Initializer::RandomUniform RandomUniform;

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
	Layer* Initialize(Initializer initializer);
};

struct Optimizer {
	struct SGD {
		double decay;
		double learning_rate;

		SGD(double learning_rate, double decay = 0) {
			this->decay = decay;
			this->learning_rate = learning_rate;
		}
	};
	struct Momentum {
		double decay;
		double learning_rate;
		double momentum;

		Momentum(double learning_rate, double momentum, double decay = 0) {
			this->decay = decay;
			this->learning_rate = learning_rate;
			this->momentum = momentum;
		}
	};
	struct Nesterov {
		double decay;
		double learning_rate;
		double momentum;

		Nesterov(double learning_rate, double momentum, double decay = 0) {
			this->decay = decay;
			this->learning_rate = learning_rate;
			this->momentum = momentum;
		}
	};

	int type;

	float *gradient;

	double decay;
	double learning_rate;
	double momentum;

	void Construct(int type, double learning_rate, double momentum, double decay, int number_parameters);

	Optimizer(int type, double learning_rate, double momentum, double decay, int number_parameters);
	Optimizer(SGD SGD);
	Optimizer(Momentum Momentum);
	Optimizer(Nesterov Nesterov);
	~Optimizer();

	double Calculate_Gradient(int index, double gradient, int iterations = 0);

	Optimizer* Copy(int number_parameters);
};

typedef Optimizer::SGD SGD;

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
	void Load_Weights(string path);
	void Save_Weights(string path);
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