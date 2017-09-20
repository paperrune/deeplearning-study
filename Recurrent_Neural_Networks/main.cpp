#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "RNN.h"

void Read_MNIST(char training_set_images[], char training_set_labels[], char test_set_images[], char test_set_labels[], int time_step, int number_training, int number_test, double ***input, double ***target_output){
	FILE *file;

	if(file = fopen(training_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char pixel;

			for(int j = 0;j < time_step;j++){
				for(int k = 0;k < 28 * 28 / time_step;k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", training_set_images);
	}

	if(file = fopen(training_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = 0;h < number_training;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int j = 0;j < 10;j++){
				target_output[h][time_step - 1][j] = (j == label);
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", training_set_labels);
	}

	if(file = fopen(test_set_images, "rb")){
		for(int h = 0, value;h < 4;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char pixel;

			for(int j = 0;j < time_step;j++){
				for(int k = 0;k < 28 * 28 / time_step;k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", test_set_images);
	}

	if(file = fopen(test_set_labels, "rb")){
		for(int h = 0, value;h < 2;h++){
			fread(&value, sizeof(int), 1, file);
		}
		for(int h = number_training;h < number_training + number_test;h++){
			unsigned char label;

			fread(&label, sizeof(unsigned char), 1, file);

			for(int j = 0;j < 10;j++){
				target_output[h][time_step - 1][j] = (j == label);
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", test_set_labels);
	}
}

int main(){
	char *type_layer[] = {"MNIST", "Clstm", "Lce,sm"};

	int batch_size		 = 60;
	int time_step		 = 28;
	int time_stride		 = 28;
	int number_map[]	 = {784 / time_step, 100, 10};
	int number_iteration = 100;
	int number_layer	 = sizeof(type_layer) / sizeof(type_layer[0]);
	int number_thread	 = 6;
	int number_training	 = 6000;
	int number_test		 = 1000;

	/* Training using the entire dataset takes about 800 seconds per iteration on the i7-4790K with 6 threads.
	int number_training	 = 60000;
	int number_test		 = 10000;
	*/

	int *length_data = new int[number_training + number_test];
	int *output_mask = new int[time_step];

	double epsilon				= 0.001;
	double gradient_threshold	= 10;
	double learning_rate		= 0.005; // 0.001 for vanilla RNN

	double ***input			= new double**[number_training + number_test];
	double ***target_output	= new double**[number_training + number_test];

	Recurrent_Neural_Networks *RNN = new Recurrent_Neural_Networks(type_layer, number_layer, 0, number_map);

	for(int h = 0;h < number_training + number_test;h++){
		length_data[h] = time_step;

		input[h]		 = new double*[length_data[h]];
		target_output[h] = new double*[length_data[h]];

		for(int t = 0;t < length_data[h];t++){
			input[h][t]			= new double[number_map[0]];
			target_output[h][t] = new double[number_map[number_layer - 1]];
		}
	}
	for(int t = 0;t < time_step;t++){
		output_mask[t] = (t == time_step - 1);
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", time_step, number_training, number_test, input, target_output);	

	RNN->Initialize_Parameter(0, 0.2, -0.1);
	omp_set_num_threads(number_thread);

	for(int h = 0, time = clock();h < number_iteration;h++){
		int number_correct[2] = {0, };

		double loss = RNN->Train(batch_size, number_training, time_step, time_stride, length_data, output_mask, epsilon, gradient_threshold, learning_rate, input, target_output);

		double *output = new double[number_map[number_layer - 1]];

		for(int i = number_training;i < number_training + number_test;i++){
			for(int t = 0;t < time_step;t++){
				int argmax;

				double max = 0;

				RNN->Test(t == 0, input[i][t], output);

				if(output_mask == 0 || output_mask[t]){
					for(int j = 0;j < number_map[number_layer - 1];j++){
						if(max < output[j]){
							argmax = j;
							max = output[j];
						}
					}
					number_correct[(i < number_training) ? (0):(1)] += (int)target_output[i][t][argmax];
				}
			}
		}
		printf("score: %d / %d, %d / %d  loss: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, loss, h + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
		learning_rate *= 0.977;

		delete[] output;
	}

	for(int h = 0;h < number_training + number_test;h++){
		for(int t = 0;t < length_data[h];t++){
			delete[] input[h][t];
			delete[] target_output[h][t];
		}		
		delete[] input[h];
		delete[] target_output[h];
	}
	delete[] input;
	delete[] target_output;
	delete[] length_data;
	delete[] output_mask;
	delete RNN;

	return 0;
}
