#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "../Neural_Networks.h"

void Read_CIFAR_10(string path, int number_training, int number_test, float **input, float **target_output) {
	ifstream file;

	string filename[] = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin" };

	for (int h = 0; h < 6; h++) {
		file.open(path + filename[h]);

		if (!file.is_open()) {
			cerr << "[Read_Data], " + path + filename[h] + " not found" << endl;
			return;
		}
		file.close();
	}
	for (int h = 0, index = 0; h < number_training; h++) {
		unsigned char value;

		if (h % 10000 == 0) {
			if (file.is_open()) {
				file.close();
			}
			file.open(path + filename[index++], ifstream::binary);
		}
		file.read((char*)(&value), 1);

		for (int j = 0; j < 10; j++) {
			target_output[h][j] = (j == value);
		}
		for (int j = 0; j < 3 * 32 * 32; j++) {
			file.read((char*)&value, 1);
			input[h][j] = value / 255.0;
		}
	}
	file.close();
	file.open(path + filename[5], ifstream::binary);

	for (int h = number_training; h < number_training + number_test; h++) {
		unsigned char value;

		file.read((char*)(&value), 1);

		for (int j = 0; j < 10; j++) {
			target_output[h][j] = (j == value);
		}
		for (int j = 0; j < 3 * 32 * 32; j++) {
			file.read((char*)&value, 1);
			input[h][j] = value / 255.0;
		}
	}
	file.close();
}

int main() {
	int batch_size = 100;
	int number_iterations = 100;
	int number_training = 50000;
	int number_test = 10000;

	float **input = new float*[number_training + number_test];
	float **target_output = new float*[number_training + number_test];

	double epsilon = 0.001;
	double decay_rate = 0.977;
	double learning_rate = 0.001;

	string path;

	vector<int> number_nodes = { 3 * 32 * 32, 10 };

	vector<Layer*> layer;

	// train from scratch *****************
	Neural_Networks NN = Neural_Networks();

	// layer[0] ~ layer[10]
	layer.push_back(NN.Add(new Layer("CIFAR-10", 3, 32, 32)));
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		16, 32, 32)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	16, 32, 32)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		16, 32, 32)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	16, 32, 32)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		16, 32, 32)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	16, 32, 32)));	// ReLU activation

	// layer[11] ~ layer[19]
	layer.push_back(NN.Add(new Layer("BN,ReLU", 32, 16, 16)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		32, 16, 16)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	32, 16, 16)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 32, 16, 16)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		32, 16, 16)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	32, 16, 16)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 32, 16, 16)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		32, 16, 16)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	32, 16, 16)));	// ReLU activation

	// layer[20] ~ layer[28]
	layer.push_back(NN.Add(new Layer("BN,ReLU", 64, 8, 8)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		64, 8, 8)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	64, 8, 8)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 64, 8, 8)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		64, 8, 8)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	64, 8, 8)));	// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 64, 8, 8)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN",		64, 8, 8)));	// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU",	64, 8, 8)));	// ReLU activation

	// layer[29] ~ layer[30]
	layer.push_back(NN.Add(new Layer("", 64, 1, 1)));
	layer.push_back(NN.Add(new Layer("CE,softmax", 10)));		// cross-entropy loss, softmax activation

	int l = 0;

	// layer[1] ~ layer[10]
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");

	// layer[11] ~ layer[19]
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3),stride(2x2)");	// 3x3 convolution with 2x2 stride
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "W,kernel(1x1),stride(2x2)");	// projection shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");

	// layer[20] ~ layer[28]
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3),stride(2x2)");	// 3x3 convolution with 2x2 stride
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "W,kernel(1x1),stride(2x2)");	// projection shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-1], "W,kernel(3x3)");				// 3x3 convolution
	layer[++l]->Connect(layer[l-3], "add");							// shortcut
	layer[l]->Connect(layer[l-1], "add");

	// layer[29] ~ layer[30]
	layer[++l]->Connect(layer[l-1], "P,average");					// global average pooling
	layer[++l]->Connect(layer[l-1], "W");							// fully connected

	srand(0); NN.Initialize(0.01);
	// ************************************

	// or load pretrained model
	// Neural_Networks NN = Neural_Networks("CIFAR-10_resnet-20.txt");

	/*int number_threads;

	cout << "The number of threads : ";
	cin >> number_threads;
	cin.ignore();

	omp_set_num_threads(number_threads);*/

	cout << "path where CIFAR-10 dataset is : ";
	getline(cin, path);

	for (int h = 0; h < number_training + number_test; h++) {
		input[h] = new float[number_nodes.front()];
		target_output[h] = new float[number_nodes.back()];
	}
	Read_CIFAR_10(path, number_training, number_test, input, target_output);

	for (int g = 0, time = clock(); g < number_iterations; g++) {
		int score[2] = { 0, };

		float **output = new float*[batch_size];

		double loss = NN.Train(batch_size, number_training, input, target_output, learning_rate, epsilon);

		for (int h = 0; h < batch_size; h++) {
			output[h] = new float[number_nodes.back()];
		}
		for (int i = 0, h = 0; i < number_training + number_test; i++) {
			if (++h == batch_size || i == number_training + number_test - 1) {
				NN.Test(h, &input[i - h + 1], output);

				for (int g = 0, argmax; g < h; g++) {
					double max = 0;

					for (int j = 0; j < number_nodes.back(); j++) {
						if (max < output[g][j]) {
							max = output[g][argmax = j];
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
