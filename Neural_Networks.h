#ifndef Neural_Networks_H
#define Neural_Networks_H

#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

struct Index;
struct Initializer;
struct Layer;
struct Loss;
struct LSTM;
struct Optimizer;
struct RNN;

struct Batch_Normalization {
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
	void Destruct();
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
	int type;

	float *weight;

	string properties;

	Initializer *initializer;

	Layer *layer;
	Layer *parent_layer;

	Optimizer *optimizer;

	vector<int> *time_connection[2];

	vector<Index> *from_error;
	vector<Index> *from_neuron;
	vector<Index> *from_weight;

	Connection(Layer *layer, Layer *parent_layer, string properties, unordered_multimap<int, int> *time_connection, int type = 0);
	~Connection();

	void Destruct();
	void Initialize();
	void Optimizer(Optimizer *optimizer);

	Connection* Copy(int type = 0);
	Connection* Initializer(Initializer initializer);
};

struct Dropout {
	bool *mask;

	int batch_size;
	int number_nodes;

	double rate;

	Dropout(int number_nodes, double rate);
	~Dropout();

	void Initialize_Mask(int seed = -1);
	void Resize_Memory(int batch_size);
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

	void Destruct();
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

	Dropout *dropout;

	Initializer *initializer;

	LSTM *lstm;

	Optimizer *optimizer;

	RNN *rnn;

	Layer(int number_maps, string properties = "");
	Layer(int time_step, int number_maps, string properties = "");
	Layer(int time_step, int number_maps, int map_width, string properties = "");
	Layer(int time_step, int number_maps, int map_width, int map_height, string properties = "");
	Layer(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties = "");
	~Layer();

	void Activate(int time_index, bool training = false);
	void Adjust_Parameter(int iterations);
	void Backward(int time_index);
	void Compile(Optimizer *optimizer);
	void Construct();
	void Destruct();
	void Differentiate(int time_index, Loss *loss = nullptr, float **y_batch = nullptr);
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

struct Loss {
	enum { connectionist_temporal_classification, cross_entropy, mean_squared_error };

	struct CTC {
		int number_labels;

		string blank, *label;

		unordered_map<string, int> label_index;

		CTC(int number_labels, string label[], string blank = "");
		~CTC();

		void Decode(vector<string> &hypothesis, int sequence_length, float likelihood[]);

		int Search_Label(string label);

		double Backward_Algorithm(vector<string> &reference, int sequence_length, float _likelihood[], double **beta);
		double Calculate_Error(vector<string> &reference, int sequence_length, float *error, float *likelihood);
		double Forward_Algorithm(vector<string> &reference, int sequence_length, float _likelihood[], double **alpha);
		double Log_Add(double a, double b);

		CTC *Copy();
	};

	int type;

	CTC *ctc;

	Loss(int type) {
		this->ctc = nullptr;
		this->type = type;
	}
	Loss(CTC ctc) {
		this->ctc = ctc.Copy();
		this->type = connectionist_temporal_classification;
		ctc.label = nullptr;
	}
	~Loss() {}

	void Destruct() {
		if (ctc) {
			delete ctc;
		}
	}

	Loss* Copy() {
		Loss *loss = new Loss(type);

		if (type == connectionist_temporal_classification) {
			loss->ctc = ctc->Copy();
		}
		return loss;
	}
};

typedef Loss::CTC CTC;

struct LSTM {
	enum { forget = 0, input = 1, output = 2, cell = 3, cell_output = 4, number_node_types = 5, number_weight_types = 4 };

	bool *time_mask;

	int activation;
	int batch_size;
	int direction;
	int map_width;
	int map_height;
	int map_depth;
	int map_size;
	int number_maps;
	int number_nodes;
	int recurrent_activation;
	int time_step;

	float *bias[number_weight_types];
	float *error[number_node_types][2];
	float *neuron[number_node_types][2];

	string properties;

	Batch_Normalization *batch_normalization[number_node_types][2];

	Initializer *initializer[number_weight_types];

	Layer *layer;

	Optimizer *optimizer[number_weight_types];

	LSTM(int time_step, int number_nodes, string properties = "", Layer *layer = nullptr);
	LSTM(int time_step, int number_maps, int map_width, string properties = "", Layer *layer = nullptr);
	LSTM(int time_step, int number_maps, int map_width, int map_height, string properties = "", Layer *layer = nullptr);
	LSTM(int time_step, int number_maps, int map_width, int map_height, int map_depth, string properties = "", Layer *layer = nullptr);
	~LSTM();

	void Activate(int time_index, bool training);
	void Adjust_Parameter(int iterations);
	void Backward(int time_index);
	void Compile(Optimizer *optimizer);
	void Construct(Layer *layer);
	void Destruct();
	void Differentiate(int time_index);
	void Forward(int time_index);
	void Initialize();
	void Optimizer(Optimizer *optimizer);
	void Resize_Memory(int batch_size);

	double Activation(double x, int activation);
	double Derivation(double x, int activation);

	Batch_Normalization* Batch_Normalization(double epsilon, double momentum);

	LSTM* Activation(int activation);
	LSTM* Copy(Layer *layer = nullptr);
	LSTM* Direction(int direction);
	LSTM* Initializer(Initializer initializer, int type = -1);
	LSTM* Recurrent_Activation(int activation);
	LSTM* Time_Mask(bool time_mask[], int length_mask = 0);
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
	struct Adam {
		double decay;
		double epsilon;
		double momentum[2];
		double learning_rate;

		Adam(double learning_rate, double decay = 0, double epsilon = 0.00000001, double momentum_1 = 0.9, double momentum_2 = 0.999) {
			this->decay = decay;
			this->epsilon = epsilon;
			this->learning_rate = learning_rate;
			this->momentum[0] = momentum_1;
			this->momentum[1] = momentum_2;
		}
	};
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
	float *memory[2];

	double decay;
	double epsilon;
	double learning_rate;
	double momentum[2];

	Optimizer(int type, double learning_rate, double decay, double epsilon, double momentum_1, double momentum_2, int number_parameters);
	Optimizer(SGD SGD);
	Optimizer(Momentum Momentum);
	Optimizer(Nesterov Nesterov);
	Optimizer(Adam Adam);
	~Optimizer();

	void Construct(int type, double learning_rate, double decay, double epsilon, double momentum_1, double momentum_2, int number_parameters);
	void Destruct();

	double Calculate_Gradient(int index, double gradient, int iterations);

	Optimizer* Copy(int number_parameters = 0);
};

typedef Optimizer::SGD SGD;
typedef Optimizer::Momentum Momentum;
typedef Optimizer::Nesterov Nesterov;
typedef Optimizer::Adam Adam;

struct RNN {
	bool *time_mask;

	int activation;
	int batch_size;
	int direction;
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
	void Destruct();
	void Differentiate(int time_index);
	void Forward(int time_index);
	void Initialize();
	void Optimizer(Optimizer *optimizer);
	void Resize_Memory(int batch_size);

	Batch_Normalization* Batch_Normalization(double epsilon, double momentum);

	RNN* Activation(int activation);
	RNN* Copy(Layer *layer = nullptr);
	RNN* Direction(int direction);
	RNN* Initializer(Initializer initializer);
	RNN* Time_Mask(bool time_mask[], int length_mask = 0);
};

class Neural_Networks {
private:
	int batch_size;
	int iterations;

	vector<Layer*> layer;

	Loss *loss;

	Optimizer *optimizer;

	void Resize_Memory(int batch_size);

	double Calculate_Loss(Layer *layer, float **y_batch, vector<string> label[], int sequence_length_batch[], bool training = false);
	double Evaluate(float **x_data, float **y_data, vector<string> reference[], int sequence_length[], int data_size, int batch_size);
	double Fit(float **x_train, float **y_train, vector<string> reference[], int sequence_length[], int train_size, int batch_size);
public:
	struct Activation {
		enum { linear, hard_sigmoid, relu, sigmoid, softmax, tanh };
	};

	Neural_Networks();
	~Neural_Networks();

	void Compile(Loss loss, Optimizer optimizer);
	void Load_Weights(string path);
	void Save_Weights(string path);
	void Predict(float input[], float output[]);
	void Predict(float **input, float **output, int batch_size = 1);

	int* Shuffle(int *data, int data_size, int seed = 0);

	float** Shuffle(float **data, int data_size, int seed = 0);

	double Evaluate(float **x_data, float **y_data, int data_size, int batch_size = 1);
	double Evaluate(float **x_data, vector<string> hypothesis[], int sequence_length[], int data_size, int batch_size = 1);
	double Fit(float **x_train, float **y_train, int train_size, int batch_size = 1);
	double Fit(float **x_train, vector<string> reference[], int sequence_length[], int train_size, int batch_size = 1);

	vector<string>* Shuffle(vector<string> *data, int data_size, int seed = 0);

	Connection* Connect(int from, int to, string properties, unordered_multimap<int, int> *time_connection = nullptr);

	Layer* Add(int number_nodes, string properties = "");
	Layer* Add(int number_maps, int map_width, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, string properties = "");
	Layer* Add(int number_maps, int map_width, int map_height, int map_depth, string properties = "");
	Layer* Add(Layer layer);

	LSTM* Add(LSTM LSTM);

	RNN* Add(RNN RNN);
};

typedef Neural_Networks::Activation Activation;

#endif