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
	int batch_size = 50;
	int number_iterations = 100;
	int number_threads;
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

	// layer[0] ~ layer[7]
	layer.push_back(NN.Add(new Layer("CIFAR-10", 3, 32, 32)));
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation	
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 16, 32, 32)));		// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 16, 32, 32)));		// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 16, 32, 32)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 16, 32, 32)));		// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 16, 32, 32)));		// ReLU activation

	// layer[8] ~ layer[13]
	layer.push_back(NN.Add(new Layer("BN,ReLU", 32, 16, 16)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 32, 16, 16)));		// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 32, 16, 16)));		// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 32, 16, 16)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 32, 16, 16)));		// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 32, 16, 16)));		// ReLU activation

	// layer[14] ~ layer[19]
	layer.push_back(NN.Add(new Layer("BN,ReLU", 64, 8, 8)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 64, 8, 8)));			// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 64, 8, 8)));		// ReLU activation
	layer.push_back(NN.Add(new Layer("BN,ReLU", 64, 8, 8)));	// batch normalization, ReLU activation
	layer.push_back(NN.Add(new Layer("BN", 64, 8, 8)));			// batch normalization
	layer.push_back(NN.Add(new Layer("ReLU", 64, 8, 8)));		// ReLU activation

	// layer[20] ~ layer[21]
	layer.push_back(NN.Add(new Layer("", 64, 1, 1)));
	layer.push_back(NN.Add(new Layer("CE,softmax", 10)));		// cross-entropy loss, softmax activation

	layer[1]->Connect(layer[0], "W,kernel(3x3)");				// 3x3 convolution
	layer[2]->Connect(layer[1], "W,kernel(3x3)");				// 3x3 convolution
	layer[3]->Connect(layer[2], "W,kernel(3x3)");				// 3x3 convolution
	layer[4]->Connect(layer[1], "add");							// shortcut
	layer[4]->Connect(layer[3], "add");
	layer[5]->Connect(layer[4], "W,kernel(3x3)");				// 3x3 convolution
	layer[6]->Connect(layer[5], "W,kernel(3x3)");				// 3x3 convolution
	layer[7]->Connect(layer[4], "add");							// shortcut
	layer[7]->Connect(layer[6], "add");

	layer[8]->Connect(layer[7],   "W,kernel(3x3),stride(2x2)");	// 3x3 convolution with 2x2 stride
	layer[9]->Connect(layer[8],   "W,kernel(3x3)");				// 3x3 convolution
	layer[10]->Connect(layer[7],  "W,kernel(1x1),stride(2x2)");	// projection shortcut
	layer[10]->Connect(layer[9],  "add");
	layer[11]->Connect(layer[10], "W,kernel(3x3)");				// 3x3 convolution
	layer[12]->Connect(layer[11], "W,kernel(3x3)");				// 3x3 convolution
	layer[13]->Connect(layer[10], "add");						// shortcut
	layer[13]->Connect(layer[12], "add");

	layer[14]->Connect(layer[13], "W,kernel(3x3),stride(2x2)");	// 3x3 convolution with 2x2 stride
	layer[15]->Connect(layer[14], "W,kernel(3x3)");				// 3x3 convolution
	layer[16]->Connect(layer[13], "W,kernel(1x1),stride(2x2)"); // projection shortcut
	layer[16]->Connect(layer[15], "add");
	layer[17]->Connect(layer[16], "W,kernel(3x3)");				// 3x3 convolution
	layer[18]->Connect(layer[17], "W,kernel(3x3)");				// 3x3 convolution
	layer[19]->Connect(layer[16], "add");						// shortcut
	layer[19]->Connect(layer[18], "add");

	layer[20]->Connect(layer[19], "P,average");					// global average pooling
	layer[21]->Connect(layer[20], "W");							// fully connected

	srand(0); NN.Initialize(0.01);
	// ************************************

	// or load pretrained model
	// Neural_Networks NN = Neural_Networks("CIFAR-10_resnet-20.txt");

#ifndef Neural_Networks_CUDA_H
	cout << "The number of threads : ";
	cin >> number_threads;
	cin.ignore();

	omp_set_num_threads(number_threads);
#endif

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
