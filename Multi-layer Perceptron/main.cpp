#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "MLP.h"

void Read_MNIST(char training_set_images[], char training_set_labels[], char test_set_images[], char test_set_labels[], int number_training, int number_test, double **input, double **target_output){
	FILE *file;

	if(file = fopen(training_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char pixel;

			for(int j = 0;j < 28 * 28;j++){
				fread(&pixel, sizeof(unsigned char), 1, file);
				input[h][j] = pixel / 255.0;
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", training_set_images);
	}

	if(file = fopen(training_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int j = 0;j < 10;j++){
				target_output[h][j] = (j == label);
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", training_set_labels);
	}

	if(file = fopen(test_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char pixel;

			for(int j = 0;j < 28 * 28;j++){
				fread(&pixel, sizeof(unsigned char), 1, file);
				input[h][j] = pixel / 255.0;
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", test_set_images);
	}

	if(file = fopen(test_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int j = 0;j < 10;j++){
				target_output[h][j] = (j == label);
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found.\n", test_set_labels);
	}
}

int main(){
	char *type_layer[] = {"MNIST", "Cbn", "Cbn,do.5", "Lce,sm"};

	int batch_size		 = 60;
	int number_iteration = 100;
	int number_layer	 = sizeof(type_layer) / sizeof(type_layer[0]);
	int number_neuron[]	 = {28 * 28, 400, 400, 10};
	int number_thread	 = 4;
	int number_training	 = 60000;
	int number_test		 = 10000;

	double epsilon		 = 0.001;
	double learning_rate = 0.005;

	double **input			= new double*[number_training + number_test];
	double **target_output	= new double*[number_training + number_test];

	Multilayer_Perceptron *MLP = new Multilayer_Perceptron(type_layer, number_layer, number_neuron);

	for(int h = 0;h < number_training + number_test;h++){
		input[h]		 = new double[number_neuron[0]];
		target_output[h] = new double[number_neuron[number_layer - 1]];
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", number_training, number_test, input, target_output);

	MLP->Initialize_Parameter(0, 0.2, -0.1);
	omp_set_num_threads(number_thread);

	for(int h = 0, time = clock();h < number_iteration;h++){
		int number_correct[2] = {0, };			

		double loss = MLP->Train(batch_size, number_training, epsilon, learning_rate, input, target_output);

		double *output = new double[number_neuron[number_layer - 1]];

		for(int i = 0;i < number_training + number_test;i++){
			int argmax;

			double max = 0;

			MLP->Test(input[i], output);

			for(int j = 0;j < number_neuron[number_layer - 1];j++){
				if(max < output[j]){
					argmax = j;
					max = output[j];
				}
			}
			number_correct[(i < number_training) ? (0):(1)] += (int)target_output[i][argmax];
		}
		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, loss, h + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
		learning_rate *= 0.993;

		delete[] output;
	}

	for(int h = 0;h < number_training + number_test;h++){
		delete[] input[h];
		delete[] target_output[h];
	}
	delete[] input;
	delete[] target_output;
	delete MLP;

	return 0;
}