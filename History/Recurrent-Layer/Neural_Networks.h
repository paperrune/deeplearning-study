#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <vector>

using namespace std;

struct Index;
struct Initializer;
struct Layer;
struct Optimizer;
struct RNN;

namespace Activation {
	enum { linear, relu, sigmoid, softmax, tanh };
}
namespace Loss {
	enum { cross_entropy, mean_squared_error };
}

class Batch_Normalization {
public:
	int batch_size;
	int map_size;
	int number_maps;
	int number_nodes;
	int time_step;

	float *gamma;
	float *beta;
	float *mean;
	float *variance;
	float *moving_mean;
	float *moving_variance;

	float *error_backup;
	float *error_normalized;
	float *neuron_backup;
	float *neuron_normalized;

	double epsilon;
	double momentum;

	Initializer *gamma_initializer;
	Initializer *beta_initializer;
	Initializer *moving_mean_initializer;
	Initializer *moving_variance_initializer;

	Layer *layer;

	Optimizer *gamma_optimizer;
	Optimizer *beta_optimizer;

	Batch_Normalization(int time_step, int number_maps, int map_size, double epsilon, double momentum, Layer *layer = nullptr);
	~Batch_Normalization();

	void Activate(int time_index, float neuron[], bool training);
	void Adjust_Parameter(int iterations);
	void Differentiate(int time_index, float error[]);
	void Initialize();
	void Load(ifstream &file);
	void Optimizer(Optimizer &optimizer);
	void Resize_Memory(int batch_size);
	void Save(ofstream &file);

	Batch_Normalization* Beta_Initializer(Initializer initializer);
	Batch_Normalization* Copy();
	Batch_Normalization* Gamma_Initializer(Initializer initializer);
	Batch_Normalization* Moving_Mean_Initializer(Initializer initializer);
	Batch_Normalization* Moving_Variance_Initializer(Initializer initializer);
};


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

	Initializer *initializer;

	Layer *layer;
	Layer *parent_layer;

	Optimizer *optimizer;

	vector<Index> *from_error;
	vector<Index> *from_neuron;
	vector<Index> *from_weight;

	Connection(Layer *layer, Layer *parent_layer, string properties);
	~Connection();

	void Initialize();
	void Optimizer(Optimizer *optimizer);

	Connection* Initializer(Initializer initializer);
};

struct Index {
	int next_node;
	int prev_node;
	int weight;
};

struct Initializer {
	struct GlorotNormal {
		int seed;

		default_random_engine *generator;

		GlorotNormal(int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct GlorotUniform {
		int seed;

		default_random_engine *generator;

		GlorotUniform(int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct HeNormal {
		int seed;

		default_random_engine *generator;

		HeNormal(int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct HeUniform {
		int seed;

		default_random_engine *generator;

		HeUniform(int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
		}
	};
	struct Orthogonal {
		int seed;

		double gain;

		default_random_engine *generator;

		Orthogonal(double gain = 1.0, int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
			this->gain = gain;
		}
	};
	struct RandomNormal {
		int seed;

		double mean;
		double stdv;

		default_random_engine *generator;

		RandomNormal(double stdv, double mean = 0, int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
			this->mean = mean;
			this->stdv = stdv;
		}
	};
	struct RandomUniform {
		int seed;

		double max;
		double min;

		default_random_engine *generator;

		RandomUniform(double min, double max, int seed = -1) {
			generator = ((this->seed = seed) >= 0) ? (new default_random_engine(seed)) : (new default_random_engine(rand()));
			this->max = max;
			this->min = min;
		}
	};

	int seed;
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
	Initializer(Orthogonal initializer);
	Initializer(RandomNormal initializer);
	Initializer(RandomUniform initializer);
	~Initializer();

	void Random(int memory_size, float memory[], int fan_in, int fan_out);

	Initializer* Copy();
};

typedef Initializer::GlorotNormal GlorotNormal;
typedef Initializer::GlorotUniform GlorotUniform;
typedef Initializer::HeNormal HeNormal;
typedef Initializer::HeUniform HeUniform;
typedef Initializer::Orthogonal Orthogonal;
typedef Initializer::RandomNormal RandomNormal;
typedef Initializer::RandomUniform RandomUniform;

struct Layer {
	bool *dropout_mask;
	bool *time_mask;

	int activation;
	int batch_size;
	int map_width;
	int map_height;
	int map_depth;
	int map_size;
	int number_maps;
	int number_nodes;
	int time_step;

	float *bias;
	float *error;
	float *neuron;

	string properties;

	vector<Connection*> connection;
	vector<Connection*> child_connection;

	Batch_Normalization *batch_normalization;

	Initializer *initializer;

	Optimizer *optimizer;

	RNN *RNN;

	Layer(int time_step, int number_maps, int map_width = 1, int map_height = 1, int map_depth = 1, string properties = "", ::RNN *RNN = nullptr);
	~Layer();

	void Activate(int time_index, bool training = false);
	void Adjust_Parameter(int iterations);
	void Backward(int time_index);
	void Compile(Optimizer *optimizer);
	void Differentiate(int time_index, int loss = -1, float **y_batch = nullptr);
	void Forward(int time_index);
	void Initialize();
	void Optimizer(Optimizer *optimizer);
	void Resize_Memory(int batch_size);

	Batch_Normalization* Batch_Normalization(double epsilon = 0.001, double momentum = 0.99);

	Connection* Search_Child_Connection(string properties);
	Connection* Search_Connection(string properties);

	Layer* Activation(int activation);
	Layer* Copy();
	Layer* Initializer(Initializer initializer);
	Layer* Time_Mask(bool time_mask[], int length_mask = 0);
};

struct Matrix {
	int number_rows;
	int number_columns;

	double *data;

	Matrix(int number_rows = 0, int number_columns = 0);
	~Matrix();

	Matrix(const Matrix& matrix) : Matrix(matrix.number_rows, matrix.number_columns) {
		for (int i = 0; i < number_rows; i++) {
			for (int j = 0; j < number_columns; j++) {
				(*this)(i, j) = matrix(i, j);
			}
		}
	}

	double& operator() (int y, int x) {
		return data[x + number_columns * y];
	}
	double operator() (int y, int x) const {
		return data[x + number_columns * y];
	}

	Matrix& operator= (const Matrix &matrix) {
		if (this != &matrix) {
			number_columns = matrix.number_columns;
			number_rows = matrix.number_rows;
			memcpy(data = (double*)realloc(data, sizeof(double) * number_rows * number_columns), matrix.data, sizeof(double) * number_rows * number_columns);
		}
		return *this;
	}
	Matrix operator* (const Matrix &matrix) {
		return Multiplication(*this, matrix);
	}

	void Gram_Schmidt_Process(double gain);
	void Identity();
	void LQ_Decomposition(Matrix &L, Matrix &Q);
	void QR_Decomposition(Matrix &Q, Matrix &R);
	void Transpose();

	Matrix Multiplication(const Matrix &A, const Matrix &B);
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

	double Calculate_Gradient(int index, double gradient, int iterations);

	Optimizer* Copy(int number_parameters);
};

typedef Optimizer::SGD SGD;
typedef Optimizer::Momentum Momentum;
typedef Optimizer::Nesterov Nesterov;

struct RNN {
	bool *time_mask;

	int activation;
	int batch_size;
	int map_width;
	int map_height;
	int map_depth;
	int map_size;
	int number_maps;
	int number_nodes;
	int time_step;

	float *bias;
	float *error[2];
	float *neuron[2];

	string properties;

	Batch_Normalization *batch_normalization[2];

	Initializer *initializer;

	Layer *layer;

	Optimizer *optimizer;

	RNN(int time_step, int number_nodes, string properties = "", Layer *layer = nullptr);
	RNN(int time_step, int number_maps, int map_width, string properties = "", Layer *layer = nullptr);
	RNN(int time_step, int number_maps, int map_width, int map_height, string properties = "", Layer *layer = nullptr);
	RNN(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties = "", Layer *layer = nullptr);
	~RNN();

	void Activate(int time_index, bool training);
	void Adjust_Parameter(int iterations);
	void Backward(int time_index);
	void Compile(Optimizer *optimizer);
	void Construct(Layer *layer);
	void Differentiate(int time_index);
	void Forward(int time_index);
	void Initialize();
	void Optimizer(Optimizer *optimizer);
	void Resize_Memory(int batch_size);

	Batch_Normalization* Batch_Normalization(double epsilon, double momentum);

	RNN* Activation(int activation);
	RNN* Copy(Layer *layer = nullptr);
	RNN* Initializer(Initializer initializer);
	RNN* Time_Mask(bool time_mask[], int length_mask = 0);
};

class Neural_Networks {
private:
	int batch_size;
	int iterations;
	int loss;

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

	float** Shuffle(float **data, int data_size, int seed = 0);

	double Evaluate(float **x_data, float **y_data, int data_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);

	Connection* Connect(int from, int to, string properties);

	Layer* Add(int number_nodes, string properties = "");
	Layer* Add(int number_maps, int map_width, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, int map_depth, string properties = "");
	Layer* Add(Layer layer);

	RNN* Add(RNN RNN);
};

#endif
