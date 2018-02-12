#include <fstream>
#include <iostream>
#include <omp.h>
#include <time.h>

#include "Neural_Networks.h"

void Read_MNIST(string training_set_images, string training_set_labels, string test_set_images, string test_set_labels, int number_training, int number_test, double **input, double **target_output) {
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
	bool *output_mask;

	// string type_layer[] = { "MNIST", "Cbn,rc", "Lce,sm" };
	string type_layer[] = { "MNIST", "Cbn,lstm", "Lce,sm" };

	int batch_size = 60;
	int number_maps[] = { 28, 100, 10 };
	int number_iterations = 100;
	int number_layers = sizeof(type_layer) / sizeof(type_layer[0]);
	int number_threads = 4;
	int number_training = 60000;
	int number_test = 10000;
	int time_step = 28;

	double epsilon = 0.001;
	double learning_rate = 0.005; // 0.001 for vanilla recurrent
	double noise_variance = 0.001;
	double decay_rate = 0.977;

	double **input = new double*[number_training + number_test];
	double **target_output = new double*[number_training + number_test];

	Neural_Networks NN = Neural_Networks(type_layer, number_layers, number_maps, nullptr, nullptr, time_step);

	for (int h = 0; h < number_training + number_test; h++) {
		input[h] = new double[time_step * number_maps[0]];
		target_output[h] = new double[time_step * number_maps[number_layers - 1]];
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", number_training, number_test, input, target_output);

	NN.Initialize_Parameter(0, 0.1, 0.1);
	omp_set_num_threads(number_threads);

	for (int h = 0; h < number_training + number_test; h++) {
		memcpy(&target_output[h][(time_step - 1) * number_maps[number_layers - 1]], target_output[h], sizeof(double) * number_maps[number_layers - 1]);
	}
	memset(output_mask = new bool[time_step], 0, sizeof(bool) * time_step);
	output_mask[time_step - 1] = true;

	for (int h = 0, time = clock(); h < number_iterations; h++) {
		int number_correct[2] = { 0, };

		double loss = NN.Train(batch_size, number_training, epsilon, learning_rate, input, target_output, output_mask, noise_variance);

		double **output = new double*[batch_size];

		for (int h = 0; h < batch_size; h++) {
			output[h] = new double[time_step * number_maps[number_layers - 1]];
		}
		for (int i = 0, h = 0; i < number_training + number_test; i++) {
			if (++h == batch_size || i == number_training + number_test - 1) {
				NN.Test(h, &input[i - h + 1], output);

				for (int g = 0, argmax; g < h; g++) {
					double max = 0;

					for (int j = 0; j < time_step * number_maps[number_layers - 1]; j++) {
						if (max < output[g][j] && (output_mask == nullptr || output_mask[j / number_maps[number_layers - 1]])) {
							max = output[g][(argmax = j)];
						}
					}
					number_correct[(i - h + g + 1 < number_training) ? (0) : (1)] += (int)target_output[i - h + g + 1][argmax];
				}
				h = 0;
			}
		}
		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, loss, h + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
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
	delete[] output_mask;

	return 0;
}