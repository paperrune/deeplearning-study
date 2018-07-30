#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <time.h>

#include "Neural_Networks.h"

using namespace std;

void Read_MNIST(string training_set_images, string training_set_labels, string test_set_images, string test_set_labels, int number_training, int number_test, float **input, float **output) {
	ifstream file(training_set_images, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 4; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = 0; h < number_training; h++) {
			unsigned char pixel;

			for (int j = 0; j < 28 * 28; j++) {
				file.read((char*)(&pixel), 1);
				input[h][j] = pixel / 255.0;
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + training_set_images + " not found" << endl;
	}

	file.open(training_set_labels, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 2; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = 0; h < number_training; h++) {
			unsigned char label;

			file.read((char*)(&label), 1);

			for (int j = 0; j < 10; j++) {
				output[h][j] = (j == label);
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + training_set_labels + " not found" << endl;
	}

	file.open(test_set_images, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 4; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = number_training; h < number_training + number_test; h++) {
			unsigned char pixel;

			for (int j = 0; j < 28 * 28; j++) {
				file.read((char*)(&pixel), 1);
				input[h][j] = pixel / 255.0;
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + test_set_images + " not found" << endl;
	}

	file.open(test_set_labels, ifstream::binary);

	if (file.is_open()) {
		for (int h = 0, value; h < 2; h++) {
			file.read((char*)(&value), sizeof(int));
		}
		for (int h = number_training; h < number_training + number_test; h++) {
			unsigned char label;

			file.read((char*)(&label), 1);

			for (int j = 0; j < 10; j++) {
				output[h][j] = (j == label);
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + test_set_labels + " not found" << endl;
	}
}

int main() {
	int batch_size = 128;
	int epochs = 30;
	int number_threads = 4;
	int number_training = 60000;
	int number_test = 10000;
	int number_nodes[] = { 784, 10 };

	float **x_data = new float*[number_training + number_test];
	float **y_data = new float*[number_training + number_test];
	float **x_train = x_data;
	float **y_train = y_data;
	float **x_test = &x_data[number_training];
	float **y_test = &y_data[number_training];

	double learning_rate = 0.5;

	string path;

	Neural_Networks NN = Neural_Networks();

	cout << "path where MNIST handwritten digits dataset is : ";
	getline(cin, path);

	for (int h = 0; h < number_training + number_test; h++) {
		x_data[h] = new float[number_nodes[0]];
		y_data[h] = new float[number_nodes[1]];
	}
	Read_MNIST(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte", path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte", number_training, number_test, x_data, y_data);

	srand(0);
	omp_set_num_threads(number_threads);

	NN.Add(number_nodes[0]);
	NN.Add(512)->Activation(Activation::relu);
	NN.Add(512)->Activation(Activation::relu);
	NN.Add(number_nodes[1])->Activation(Activation::softmax);

	NN.Connect(1, 0, "W")->Initializer(RandomUniform(-0.01, 0.01));
	NN.Connect(2, 1, "W")->Initializer(RandomUniform(-0.01, 0.01));
	NN.Connect(3, 2, "W")->Initializer(RandomUniform(-0.01, 0.01));

	NN.Compile(Loss::cross_entropy, Optimizer(SGD(learning_rate)));

	for (int e = 0, time = clock(); e < epochs; e++) {
		int score[2] = { 0, };

		float **_input = new float*[batch_size];
		float **output = new float*[batch_size];

		double loss[2] = { NN.Fit(x_train, y_train, number_training, batch_size), NN.Evaluate(x_test, y_test, number_test, batch_size) };

		for (int h = 0; h < batch_size; h++) {
			output[h] = new float[number_nodes[1]];
		}
		for (int h = 0, i = 0; i < number_training + number_test; i++) {
			_input[h] = x_data[i];

			if (++h == batch_size || i == number_training + number_test - 1) {
				NN.Predict(_input, output, h);

				for (int argmax, index = i - h + 1; --h >= 0;) {
					double max = 0;

					for (int j = 0; j < number_nodes[1]; j++) {
						if (j == 0 || max < output[h][j]) {
							max = output[h][argmax = j];
						}
					}
					score[(index + h < number_training) ? (0) : (1)] += (int)y_data[index + h][argmax];
				}
				h = 0;
			}
		}
		printf("loss: %.4f / %.4f	accuracy: %.4f / %.4f	step %d  %.2f sec\n", loss[0], loss[1], 1.0 * score[0] / number_training, 1.0 * score[1] / number_test, e + 1, (double)(clock() - time) / CLOCKS_PER_SEC);

		delete[] output;
	}

	for (int i = 0; i < number_training + number_test; i++) {
		delete[] x_data[i];
		delete[] y_data[i];
	}
	delete[] x_data;
	delete[] y_data;
}
