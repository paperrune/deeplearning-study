#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "../Neural_Networks.h"

using namespace NNCPU;

void Read_MNIST(string training_set_images, string training_set_labels, string test_set_images, string test_set_labels, int number_training, int number_test, float **input, float **target_output) {
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
				target_output[h][j] = (j == label);
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
				target_output[h][j] = (j == label);
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_MNIST], " + test_set_labels + " not found" << endl;
	}
}

int main() {
	int batch_size		= 60;
	int number_iterations = 50;
	int number_threads;
	int number_training	= 60000;
	int number_test		= 10000;
	int time_step		= 28;

	float **input = new float*[number_training + number_test];
	float **target_output = new float*[number_training + number_test];

	double epsilon		 = 0.001;
	double decay_rate	 = 0.977;
	double learning_rate = 0.005;
	double noise		 = 0.001;

	string path;

	vector<int> number_nodes = { 784, 10 };

	vector<Layer*> layer;

	// train from scratch **************************
	Neural_Networks NN = Neural_Networks(time_step);

	layer.push_back(NN.Add(new Layer("MNIST", 784 / time_step)));
	layer.push_back(NN.Add(new Layer("BN,LSTM", 100)));			 // batch normalization, LSTM forward direction
	layer.push_back(NN.Add(new Layer("BN,LSTM,backward", 100))); // batch normalization, LSTM backward direction
	layer.push_back(NN.Add(new Layer("CE,softmax", 10)));		 // cross-entropy loss, softmax activation

	layer[1]->Connect(layer[0], "W");			// fully connected
	layer[1]->Connect(layer[1], "W,recurrent"); // fully connected
	layer[2]->Connect(layer[0], "W");			// fully connected
	layer[2]->Connect(layer[2], "W,recurrent"); // fully connected
	layer[3]->Connect(layer[1], "W");			// fully connected
	layer[3]->Connect(layer[2], "W");			// fully connected

	srand(0); NN.Initialize(0.1, 0.1);

	{
		bool *time_mask = new bool[time_step];

		for (int t = 0; t < time_step; t++) {
			time_mask[t] = (t == 0 || t == time_step - 1);
		}
		layer.back()->Set_Time_Mask(time_mask);
	}
	// *********************************************

	// load pretrained model
	// Neural_Networks NN = Neural_Networks("MNIST_BLSTM.txt");

	cout << "The number of threads : ";
	cin >> number_threads;
	cin.ignore();

	omp_set_num_threads(number_threads);

	cout << "path where MNIST handwritten digits dataset is : ";
	getline(cin, path);

	for (int h = 0; h < number_training + number_test; h++) {
		input[h] = new float[number_nodes.front()];
		target_output[h] = new float[time_step * number_nodes.back()];
	}
	Read_MNIST(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte", path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte", number_training, number_test, input, target_output);

	for (int h = 0; h < number_training + number_test; h++) {
		memcpy(&target_output[h][(time_step - 1) * number_nodes.back()], target_output[h], sizeof(float) * number_nodes.back());
	}
	for (int g = 0, time = clock(); g < number_iterations; g++) {
		int score[2] = { 0, };

		float **output = new float*[batch_size];

		double loss = NN.Train(batch_size, number_training, input, target_output, learning_rate, epsilon, noise);

		for (int h = 0; h < batch_size; h++) {
			output[h] = new float[time_step * number_nodes.back()];
		}
		for (int i = 0, h = 0; i < number_training + number_test; i++) {
			if (++h == batch_size || i == number_training + number_test - 1) {
				NN.Test(h, &input[i - h + 1], output);

				for (int g = 0, argmax, index = (time_step - 1) * number_nodes.back(); g < h; g++) {
					double max = 0;

					for (int j = 0; j < number_nodes.back(); j++) {
						if (max < output[g][j] + output[g][index + j]) {
							max = output[g][argmax = j] + output[g][index + j];
						}
					}
					score[(i - h + g + 1 < number_training) ? (0) : (1)] += (int)target_output[i - h + g + 1][argmax];
				}
				h = 0;
			}
		}
		printf("."); NN.Save("NN.txt");

		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", score[0], number_training, score[1], number_test, loss, g + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
		learning_rate *= decay_rate;

		for (int h = 0; h < batch_size; h++) {
			delete[] output[h];
		}
		delete[] output;
	}

	for (int h = 0; h < number_training + number_test; h++) {
		delete[] input[h];
		delete[] target_output[h];
	}
	delete[] input;
	delete[] target_output;

	return 0;
}
