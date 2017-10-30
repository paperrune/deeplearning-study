#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RNN.h"

void Recurrent_Neural_Networks::Activate(char option[], int layer_index, int time_index, int map_index){
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double ****neuron						= this->neuron[0][0][i][t];
	double ****neuron_patch[2]				= {this->neuron[0][1][i][t], this->neuron[0][2][i][t]};

	double ****reset_neuron					= this->neuron[1][0][i][t];
	double ****reset_neuron_patch[2]		= {this->neuron[1][1][i][t], this->neuron[1][2][i][t]};

	double ****update_neuron				= this->neuron[2][0][i][t];
	double ****update_neuron_patch[2]		= {this->neuron[2][1][i][t], this->neuron[2][2][i][t]};
		
	double ****forget_neuron				= this->neuron[1][0][i][t];
	double ****forget_neuron_patch[2]		= {this->neuron[1][1][i][t], this->neuron[1][2][i][t]};

	double ****input_neuron					= this->neuron[2][0][i][t];
	double ****input_neuron_patch[2]		= {this->neuron[2][1][i][t], this->neuron[2][2][i][t]};

	double ****block_cell_neuron			= this->neuron[4][0][i][t];
	double ****block_cell_neuron_patch[2]	= {this->neuron[4][1][i][t], this->neuron[4][2][i][t]};

	double ****cell_neuron					= this->neuron[5][0][i][t];
	double ****previous_cell_neuron			= this->neuron[5][1][i][(t - 1 < 0) ? (time_step):(t - 1)];

	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "bn")){
			if(strstr(type_layer[i], "gru")){
				Batch_Normalization_Activate(option, 1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 2, 2, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "lstm")){
				Batch_Normalization_Activate(option, 1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 2, 2, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 4, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 4, 2, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "rc")){
				Batch_Normalization_Activate(option, 0, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 0, 2, layer_index, time_index, map_index);
			}
			else{
				Batch_Normalization_Activate(option, 0, 0, layer_index, time_index, map_index);
			}
		}

		if(strstr(type_layer[i], "gru")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						reset_neuron[h][j][k][l]	= Sigmoid(reset_neuron[h][j][k][l] + reset_neuron_patch[0][h][j][k][l] + reset_neuron_patch[1][h][j][k][l]);
						update_neuron[h][j][k][l]	= Sigmoid(update_neuron[h][j][k][l] + update_neuron_patch[0][h][j][k][l] + update_neuron_patch[1][h][j][k][l]);
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "lstm")){
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					if(!strcmp(option, "train")){
						mask = dropout_mask[i][h][j];
					}
					else
					if(!strcmp(option, "test")){
						mask = atof(strstr(type_layer[i], "do") + 2);
					}
				}
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						forget_neuron[h][j][k][l]		= Sigmoid(forget_neuron[h][j][k][l] + forget_neuron_patch[0][h][j][k][l] + forget_neuron_patch[1][h][j][k][l]);
						input_neuron[h][j][k][l]		= Sigmoid(input_neuron[h][j][k][l] + input_neuron_patch[0][h][j][k][l] + input_neuron_patch[1][h][j][k][l]);
						block_cell_neuron[h][j][k][l]	= Tangent(block_cell_neuron[h][j][k][l] + block_cell_neuron_patch[0][h][j][k][l] + block_cell_neuron_patch[1][h][j][k][l]);						
						cell_neuron[h][j][k][l]			= forget_neuron[h][j][k][l] * previous_cell_neuron[h][j][k][l] + input_neuron[h][j][k][l] * mask * block_cell_neuron[h][j][k][l];
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "rc")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						neuron[h][j][k][l] = Tangent(neuron[h][j][k][l] + neuron_patch[0][h][j][k][l] + neuron_patch[1][h][j][k][l]);
					}
				}
			}
		}
		else{
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					if(!strcmp(option, "train")){
						mask = dropout_mask[i][h][j];
					}
					else
					if(!strcmp(option, "test")){
						mask = atof(strstr(type_layer[i], "do") + 2);
					}
				}

				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double &neuron = this->neuron[0][0][i][t][h][j][k][l];

						if(strstr(type_layer[i], "ht")){
							neuron = Tangent(neuron);
						}
						else
						if(strstr(type_layer[i], "ls")){
							neuron = Sigmoid(neuron);
						}
						else{
							neuron *= (neuron > 0);
						}

						// dropout
						neuron *= mask;
					}
				}
			}
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					double &neuron = this->neuron[0][0][i][t][h][j][k][l];

					if(strstr(type_layer[i], "ce")){
						if(strstr(type_layer[i], "sm")){
							// neuron = neuron;
						}
						else{
							neuron = 1 / (1 + exp(-neuron));
						}
					}
					else
					if(strstr(type_layer[i], "mse")){
						if(strstr(type_layer[i], "ht")){
							neuron = 2 / (1 + exp(-2 * neuron)) - 1;
						}
						else
						if(strstr(type_layer[i], "ia")){
							// neuron = neuron;
						}
						else{
							neuron = 1 / (1 + exp(-neuron));
						}
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Adjust_Parameter(int layer_index, int time_index, int map_index){
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double ****derivative						= this->derivative[0][0][i][t];
	double ****derivative_patch[2]				= {this->derivative[0][1][i][t], this->derivative[0][2][i][t]};

	double ****reset_derivative					= this->derivative[1][0][i][t];
	double ****reset_derivative_patch[2]		= {this->derivative[1][1][i][t], this->derivative[1][2][i][t]};

	double ****update_derivative				= this->derivative[2][0][i][t];
	double ****update_derivative_patch[2]		= {this->derivative[2][1][i][t], this->derivative[2][2][i][t]};

	double ****block_derivative					= this->derivative[3][0][i][t];
	double ****block_derivative_patch[2]		= {this->derivative[3][1][i][t], this->derivative[3][2][i][t]};
		
	double ****forget_derivative				= this->derivative[1][0][i][t];
	double ****forget_derivative_patch[2]		= {this->derivative[1][1][i][t], this->derivative[1][2][i][t]};

	double ****input_derivative					= this->derivative[2][0][i][t];
	double ****input_derivative_patch[2]		= {this->derivative[2][1][i][t], this->derivative[2][2][i][t]};

	double ****output_derivative				= this->derivative[3][0][i][t];
	double ****output_derivative_patch[2]		= {this->derivative[3][1][i][t], this->derivative[3][2][i][t]};

	double ****block_cell_derivative			= this->derivative[4][0][i][t];
	double ****block_cell_derivative_patch[2]	= {this->derivative[4][1][i][t], this->derivative[4][2][i][t]};
		
	double ****lower_neuron			= this->neuron[0][0][i - 1][t];
	double ****previous_neuron		= this->neuron[0][0][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double ****reset_neuron			= this->neuron[1][0][i][t];
	double ****cell_neuron_patch	= this->neuron[5][1][i][t];
	double ****previous_cell_neuron	= this->neuron[5][1][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double *forget_cell_weight	= this->cell_weight_momentum[1][i];
	double *input_cell_weight	= this->cell_weight_momentum[2][i];
	double *output_cell_weight	= this->cell_weight_momentum[3][i];

	double ****weight			= this->weight_momentum[0][i];
	double ****reset_weight		= this->weight_momentum[1][i];
	double ****update_weight	= this->weight_momentum[2][i];
		
	double ****forget_weight	= this->weight_momentum[1][i];
	double ****input_weight		= this->weight_momentum[2][i];
	double ****output_weight	= this->weight_momentum[3][i];

	if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
		if(strstr(type_layer[i], "bn")){
			if(strstr(type_layer[i], "gru")){
				Batch_Normalization_Adjust_Parameter(1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(2, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(3, 2, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "lstm")){
				Batch_Normalization_Adjust_Parameter(1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(2, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(3, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(4, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(4, 2, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(5, 0, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "rc")){
				Batch_Normalization_Adjust_Parameter(0, 1, layer_index, time_index, map_index);
				Batch_Normalization_Adjust_Parameter(0, 2, layer_index, time_index, map_index);
			}
			else{
				Batch_Normalization_Adjust_Parameter(0, 0, layer_index, time_index, map_index);
			}
		}

		if(strstr(type_layer[i], "gru")){
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, 0, 0, reset_derivative, reset_derivative_patch, 0, reset_weight);
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, 0, 0, update_derivative, update_derivative_patch, 0, update_weight);
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, 0, reset_neuron, block_derivative, block_derivative_patch, 0, weight);
		}
		else
		if(strstr(type_layer[i], "lstm")){
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, previous_cell_neuron, 0, forget_derivative, forget_derivative_patch, forget_cell_weight, forget_weight);
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, previous_cell_neuron, 0, input_derivative, input_derivative_patch, input_cell_weight, input_weight);
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, cell_neuron_patch, 0, output_derivative, output_derivative_patch, output_cell_weight, output_weight);
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, 0, 0, block_cell_derivative, block_cell_derivative_patch, 0, weight);
		}
		else
		if(strstr(type_layer[i], "rc")){
			Adjust_Parameter(layer_index, map_index, lower_neuron, previous_neuron, 0, 0, derivative, derivative_patch, 0, weight);
		}
		else{
			Adjust_Parameter(layer_index, map_index, lower_neuron, 0, 0, 0, derivative, 0, 0, weight);
		}
	}
}
void Recurrent_Neural_Networks::Adjust_Parameter(int layer_index, int map_index, double ****lower_neuron, double ****previous_neuron, double ****cell_neuron, double ****reset_neuron, double ****_derivative, double *****derivative_patch, double *cell_weight, double ****weight){
	int i = layer_index;
	int j = map_index;

	double sum = 0;

	if(lower_neuron){
		double ****derivative = (derivative_patch) ? (derivative_patch[0]):(_derivative);

		for(int m = 0;m < number_maps[i - 1];m++){
			for(int n = 0;n < kernel_height[i];n++){
				for(int o = 0;o < kernel_width[i];o++){
					double sum = 0;

					for(int h = 0;h < batch_size;h++){
						for(int k = 0;k < map_height[i];k++){
							for(int l = 0;l < map_width[i];l++){
								int index[2] = {k * stride_height[i] + n, l * stride_width[i] + o};

								if(index[0] < map_height[i - 1] && index[1] < map_width[i - 1]){
									sum += derivative[h][j][k][l] * lower_neuron[h][m][index[0]][index[1]];
								}
							}
						}
					}
					weight[j][m][n][o] -= sum;
				}
			}
		}
	}
	if(previous_neuron){			
		double ****derivative = (derivative_patch) ? (derivative_patch[1]):(_derivative);

		for(int m = 0;m < number_maps[i];m++){
			double sum = 0;

			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						sum += derivative[h][j][k][l] * previous_neuron[h][m][k][l] * ((reset_neuron) ? (reset_neuron[h][m][k][l]):(1));
					}
				}
			}
			weight[j][number_maps[i - 1] + m][0][0] -= sum;
		}
	}

	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				sum += _derivative[h][j][k][l];
			}
		}
	}
	weight[j][number_maps[i - 1] + number_maps[i]][0][0] -= sum;

	if(cell_neuron){
		double sum = 0;

		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					sum += _derivative[h][j][k][l] * cell_neuron[h][j][k][l];
				}
			}
		}
		cell_weight[j] -= sum;
	}
}
void Recurrent_Neural_Networks::Backpropagate(char option, int layer_index, int time_index, int map_index){
	if(layer_index == number_layers - 1){
		return;
	}

	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double ****derivative			 				= this->derivative[0][0][i][t];
	double ****next_derivative						= this->derivative[0][0][i][t + 1];
	double ****next_derivative_patch[2]				= {this->derivative[0][1][i][t + 1], this->derivative[0][2][i][t + 1]};

	double ****upper_derivative						= this->derivative[0][0][i + 1][t];
	double ****upper_derivative_patch[2]			= {this->derivative[0][1][i + 1][t], this->derivative[0][2][i + 1][t]};

	double ****reset_derivative						= this->derivative[1][0][i][t];
	double ****reset_derivative_patch[2]			= {this->derivative[1][1][i][t], this->derivative[1][2][i][t]};
	double ****next_reset_derivative_patch[2]		= {this->derivative[1][1][i][t + 1], this->derivative[1][2][i][t + 1]};
	double ****upper_reset_derivative_patch[2]		= {this->derivative[1][1][i + 1][t], this->derivative[1][2][i + 1][t]};

	double ****update_derivative					= this->derivative[2][0][i][t];
	double ****update_derivative_patch[2]			= {this->derivative[2][1][i][t], this->derivative[2][2][i][t]};
	double ****next_update_derivative_patch[2]		= {this->derivative[2][1][i][t + 1], this->derivative[2][2][i][t + 1]};
	double ****upper_update_derivative_patch[2]		= {this->derivative[2][1][i + 1][t], this->derivative[2][2][i + 1][t]};

	double ****block_derivative_patch[2]			= {this->derivative[3][1][i][t], this->derivative[3][2][i][t]};
	double ****next_block_derivative_patch[2]		= {this->derivative[3][1][i][t + 1], this->derivative[3][2][i][t + 1]};
	double ****upper_block_derivative_patch[2]		= {this->derivative[3][1][i + 1][t], this->derivative[3][2][i + 1][t]};

	double ****forget_derivative					= this->derivative[1][0][i][t];
	double ****forget_derivative_patch[2]			= {this->derivative[1][1][i][t], this->derivative[1][2][i][t]};
	double ****next_forget_derivative				= this->derivative[1][0][i][t + 1];
	double ****next_forget_derivative_patch[2]		= {this->derivative[1][1][i][t + 1], this->derivative[1][2][i][t + 1]};
	double ****upper_forget_derivative_patch[2]		= {this->derivative[1][1][i + 1][t], this->derivative[1][2][i + 1][t]};

	double ****input_derivative						= this->derivative[2][0][i][t];
	double ****input_derivative_patch[2]			= {this->derivative[2][1][i][t], this->derivative[2][2][i][t]};
	double ****next_input_derivative				= this->derivative[2][0][i][t + 1];
	double ****next_input_derivative_patch[2]		= {this->derivative[2][1][i][t + 1], this->derivative[2][2][i][t + 1]};
	double ****upper_input_derivative_patch[2]		= {this->derivative[2][1][i + 1][t], this->derivative[2][2][i + 1][t]};

	double ****output_derivative					= this->derivative[3][0][i][t];
	double ****next_output_derivative_patch[2]		= {this->derivative[3][1][i][t + 1], this->derivative[3][2][i][t + 1]};
	double ****upper_output_derivative_patch[2]		= {this->derivative[3][1][i + 1][t], this->derivative[3][2][i + 1][t]};

	double ****block_cell_derivative				= this->derivative[4][0][i][t];
	double ****block_cell_derivative_patch[2]		= {this->derivative[4][1][i][t], this->derivative[4][2][i][t]};
	double ****next_block_cell_derivative_patch[2]	= {this->derivative[4][1][i][t + 1], this->derivative[4][2][i][t + 1]};
	double ****upper_block_cell_derivative_patch[2]	= {this->derivative[4][1][i + 1][t], this->derivative[4][2][i + 1][t]};
		
	double ****cell_derivative						= this->derivative[5][0][i][t];
	double ****next_cell_derivative					= this->derivative[5][0][i][t + 1];		
		
	double ****previous_neuron		= this->neuron[0][0][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double ****reset_neuron			= this->neuron[1][0][i][t];
	double ****next_reset_neuron	= this->neuron[1][0][i][t + 1];

	double ****update_neuron		= this->neuron[2][0][i][t];
	double ****next_update_neuron	= this->neuron[2][0][i][t + 1];

	double ****block_neuron			= this->neuron[3][0][i][t];

	double ****forget_neuron		= this->neuron[1][0][i][t];
	double ****next_forget_neuron	= this->neuron[1][0][i][t + 1];
	double ****input_neuron			= this->neuron[2][0][i][t];
	double ****block_cell_neuron	= this->neuron[4][0][i][t];
	double ****previous_cell_neuron	= this->neuron[5][1][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double *forget_cell_weight	= this->cell_weight[1][i];
	double *input_cell_weight	= this->cell_weight[2][i];
	double *output_cell_weight	= this->cell_weight[3][i];

	double *****weight			= this->weight[0];
	double *****reset_weight	= this->weight[1];
	double *****update_weight	= this->weight[2];
		
	double *****forget_weight	= this->weight[1];
	double *****input_weight	= this->weight[2];
	double *****output_weight	= this->weight[3];

	if(option == 'A'){
		if(type_layer[i + 1][0] == 'C' || type_layer[i + 1][0] == 'L'){
			if(strstr(type_layer[i + 1], "gru")){
				Backpropagate(true,  layer_index, map_index, derivative, upper_reset_derivative_patch[0], reset_weight);
				Backpropagate(false, layer_index, map_index, derivative, upper_update_derivative_patch[0], update_weight);
				Backpropagate(false, layer_index, map_index, derivative, upper_block_derivative_patch[0], weight);
			}
			else
			if(strstr(type_layer[i + 1], "lstm")){
				Backpropagate(true,  layer_index, map_index, derivative, upper_forget_derivative_patch[0], forget_weight);
				Backpropagate(false, layer_index, map_index, derivative, upper_input_derivative_patch[0], input_weight);
				Backpropagate(false, layer_index, map_index, derivative, upper_output_derivative_patch[0], output_weight);
				Backpropagate(false, layer_index, map_index, derivative, upper_block_cell_derivative_patch[0], weight);
			}
			else
			if(strstr(type_layer[i + 1], "rc")){
				Backpropagate(true, layer_index, map_index, derivative, upper_derivative_patch[0], weight);
			}
			else{
				Backpropagate(true, layer_index, map_index, derivative, upper_derivative, weight);
			}
		}
		else
		if(type_layer[i + 1][0] == 'P'){
			if(strstr(type_layer[i + 1], "pad")){
				int margin[] = {(map_height[i + 1] - map_height[i]) / 2, (map_width[i + 1] - map_width[i]) / 2};

				for(int h = 0;h < batch_size;h++){
					for(int k = 0;k < map_height[i];k++){
						for(int l = 0;l < map_width[i];l++){
							derivative[h][j][k][l] = upper_derivative[h][j][margin[0] + k][margin[1] + l];
						}
					}
				}
			}
			else{
				int stride[] = {map_height[i] / map_height[i + 1], map_width[i] / map_width[i + 1]};

				for(int h = 0;h < batch_size;h++){
					for(int k = 0;k < map_height[i];k++){
						for(int l = 0;l < map_width[i];l++){
							derivative[h][j][k][l] = upper_derivative[h][j][k / stride[0]][l / stride[1]];
						}
					}
				}
			}
		}

		if(strstr(type_layer[i], "gru")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double sum[2] = {0, };

						for(int m = 0;m < number_maps[i];m++){
							sum[0] += next_reset_derivative_patch[1][h][m][k][l] * reset_weight[i][m][number_maps[i - 1] + j][0][0];
							sum[0] += next_update_derivative_patch[1][h][m][k][l] * update_weight[i][m][number_maps[i - 1] + j][0][0];
							sum[1] += next_block_derivative_patch[1][h][m][k][l] * weight[i][m][number_maps[i - 1] + j][0][0];
						}
						derivative[h][j][k][l] += sum[0] + sum[1] * next_reset_neuron[h][j][k][l] + (1 - next_update_neuron[h][j][k][l]) * next_derivative[h][j][k][l];
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "lstm")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double sum = 0;

						for(int m = 0;m < number_maps[i];m++){
							sum += next_forget_derivative_patch[1][h][m][k][l] * forget_weight[i][m][number_maps[i - 1] + j][0][0];
							sum += next_input_derivative_patch[1][h][m][k][l] * input_weight[i][m][number_maps[i - 1] + j][0][0];
							sum += next_output_derivative_patch[1][h][m][k][l] * output_weight[i][m][number_maps[i - 1] + j][0][0];
							sum += next_block_cell_derivative_patch[1][h][m][k][l] * weight[i][m][number_maps[i - 1] + j][0][0];
						}
						derivative[h][j][k][l] += sum;
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "rc")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double sum = 0;

						for(int m = 0;m < number_maps[i];m++){
							sum += next_derivative_patch[1][h][m][k][l] * weight[i][m][number_maps[i - 1] + j][0][0];
						}
						derivative[h][j][k][l] += sum;
					}
				}
			}
		}
	}
	else
	if(option == 'B'){
		if(strstr(type_layer[i], "gru")){
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					mask = dropout_mask[i][h][j];
				}
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double sum = 0;

						for(int m = 0;m < number_maps[i];m++){
							sum += block_derivative_patch[1][h][m][k][l] * weight[i][m][number_maps[i - 1] + j][0][0];
						}
						reset_derivative[h][j][k][l]			= sum * previous_neuron[h][j][k][l] * (1 - reset_neuron[h][j][k][l]) * reset_neuron[h][j][k][l];
						reset_derivative_patch[0][h][j][k][l]	= reset_derivative[h][j][k][l];
						reset_derivative_patch[1][h][j][k][l]	= reset_derivative[h][j][k][l];
						update_derivative[h][j][k][l]			= derivative[h][j][k][l] * (mask * block_neuron[h][j][k][l] - previous_neuron[h][j][k][l]) * (1 - update_neuron[h][j][k][l]) * update_neuron[h][j][k][l];
						update_derivative_patch[0][h][j][k][l]	= update_derivative[h][j][k][l];
						update_derivative_patch[1][h][j][k][l]	= update_derivative[h][j][k][l];
					}
				}
			}

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Differentiate(1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(2, 2, layer_index, time_index, map_index);
			}
		}
		else
		if(strstr(type_layer[i], "lstm")){
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					mask = dropout_mask[i][h][j];
				}
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double sum = 0;

						sum += next_forget_derivative[h][j][k][l] * forget_cell_weight[j];
						sum += next_input_derivative[h][j][k][l] * input_cell_weight[j];
						sum += output_derivative[h][j][k][l] * output_cell_weight[j];
						cell_derivative[h][j][k][l]	+= sum + next_forget_neuron[h][j][k][l] * next_cell_derivative[h][j][k][l];

						forget_derivative[h][j][k][l]				= cell_derivative[h][j][k][l] * previous_cell_neuron[h][j][k][l] * (1 - forget_neuron[h][j][k][l]) * forget_neuron[h][j][k][l];
						forget_derivative_patch[0][h][j][k][l]		= forget_derivative[h][j][k][l];
						forget_derivative_patch[1][h][j][k][l]		= forget_derivative[h][j][k][l];
						input_derivative[h][j][k][l]				= cell_derivative[h][j][k][l] * mask * block_cell_neuron[h][j][k][l] * (1 - input_neuron[h][j][k][l]) * input_neuron[h][j][k][l];
						input_derivative_patch[0][h][j][k][l]		= input_derivative[h][j][k][l];
						input_derivative_patch[1][h][j][k][l]		= input_derivative[h][j][k][l];
						block_cell_derivative[h][j][k][l]			= mask * cell_derivative[h][j][k][l] * input_neuron[h][j][k][l] * (1 - block_cell_neuron[h][j][k][l]) * (1 + block_cell_neuron[h][j][k][l]);
						block_cell_derivative_patch[0][h][j][k][l]	= block_cell_derivative[h][j][k][l];
						block_cell_derivative_patch[1][h][j][k][l]	= block_cell_derivative[h][j][k][l];
					}
				}
			}

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Differentiate(1, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(1, 2, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(2, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(2, 2, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(4, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(4, 2, layer_index, time_index, map_index);
			}
		}
	}
}
void Recurrent_Neural_Networks::Backpropagate(bool initialize, int layer_index, int map_index, double ****derivative, double ****upper_derivative, double *****upper_weight){
	int i = layer_index;
	int j = map_index;

	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				int ks		 = k / stride_height[i + 1];
				int ls		 = l / stride_width[i + 1];
				int index[2] = {ks - (kernel_height[i + 1] - 1), ls - (kernel_width[i + 1] - 1)};

				double sum = 0;

				if(index[0] < 0) index[0] = 0;
				if(index[1] < 0) index[1] = 0;

				for(int m = 0;m < number_maps[i + 1];m++){
					for(int n = index[0];n < map_height[i + 1] && n <= ks;n++){
						for(int o = index[1];o < map_width[i + 1] && o <= ls;o++){
							sum += upper_derivative[h][m][n][o] * upper_weight[i + 1][m][j][abs(ks - n)][abs(ls - o)];
						}
					}
				}
				derivative[h][j][k][l] = (initialize) ? (sum):(derivative[h][j][k][l] + sum);
			}
		}
	}
}
void Recurrent_Neural_Networks::Differentiate(int layer_index, int time_index, int map_index, bool output_mask[], double learning_rate, double ***target_output){
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double ****derivative					= this->derivative[0][0][i][t];
	double ****derivative_patch[2]			= {this->derivative[0][1][i][t], this->derivative[0][2][i][t]};

	double ****block_derivative				= this->derivative[3][0][i][t];
	double ****block_derivative_patch[2]	= {this->derivative[3][1][i][t], this->derivative[3][2][i][t]};

	double ****output_derivative			= this->derivative[3][0][i][t];
	double ****output_derivative_patch[2]	= {this->derivative[3][1][i][t], this->derivative[3][2][i][t]};
		
	double ****cell_derivative				= this->derivative[5][0][i][t];
		
	double ****neuron				 = this->neuron[0][0][i][t];
	double ****update_neuron		 = this->neuron[2][0][i][t];
	double ****block_neuron			 = this->neuron[3][0][i][t];

	double ****output_neuron		 = this->neuron[3][0][i][t];
	double ****cell_neuron			 = this->neuron[5][0][i][t];

	if(type_layer[i][0] == 'C'){
		if(strstr(type_layer[i], "gru")){
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					mask = dropout_mask[i][h][j];
				}
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						block_derivative[h][j][k][l]			= mask * derivative[h][j][k][l] * update_neuron[h][j][k][l] * (1 - block_neuron[h][j][k][l]) * (1 + block_neuron[h][j][k][l]);
						block_derivative_patch[0][h][j][k][l]	= block_derivative[h][j][k][l];
						block_derivative_patch[1][h][j][k][l]	= block_derivative[h][j][k][l];
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "lstm")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						cell_derivative[h][j][k][l]				= derivative[h][j][k][l] * output_neuron[h][j][k][l] * (1 - Tangent(cell_neuron[h][j][k][l])) * (1 + Tangent(cell_neuron[h][j][k][l]));
						output_derivative[h][j][k][l]			= derivative[h][j][k][l] * Tangent(cell_neuron[h][j][k][l]) * (1 - output_neuron[h][j][k][l]) * output_neuron[h][j][k][l];
						output_derivative_patch[0][h][j][k][l]	= output_derivative[h][j][k][l];
						output_derivative_patch[1][h][j][k][l]	= output_derivative[h][j][k][l];
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "rc")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						derivative[h][j][k][l]			*= (1 - neuron[h][j][k][l]) * (1 + neuron[h][j][k][l]);
						derivative_patch[0][h][j][k][l]	= derivative[h][j][k][l];
						derivative_patch[1][h][j][k][l]	= derivative[h][j][k][l];
					}
				}
			}
		}
		else{
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						double &derivative	= this->derivative[0][0][i][t][h][j][k][l];
						double neuron		= this->neuron[0][0][i][t][h][j][k][l];

						if(strstr(type_layer[i], "ht")){
							derivative *= (1 - neuron) * (1 + neuron);
						}
						else
						if(strstr(type_layer[i], "ls")){
							derivative *= (1 - neuron) * neuron;
						}
						else{
							derivative *= (neuron > 0);
						}
					}
				}
			}
		}

		if(strstr(type_layer[i], "bn")){
			if(strstr(type_layer[i], "gru")){
				Batch_Normalization_Differentiate(3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(3, 2, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "lstm")){
				Batch_Normalization_Differentiate(3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(3, 2, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(5, 0, layer_index, time_index, map_index);
			}
			else
			if(strstr(type_layer[i], "rc")){
				Batch_Normalization_Differentiate(0, 1, layer_index, time_index, map_index);
				Batch_Normalization_Differentiate(0, 2, layer_index, time_index, map_index);
			}
			else{
				Batch_Normalization_Differentiate(0, 0, layer_index, time_index, map_index);
			}
		}
	}
	else
	if(type_layer[i][0] == 'L'){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					double &derivative	= this->derivative[0][0][i][t][h][j][k][l];
					double &neuron		= this->neuron[0][0][i][t][h][j][k][l];

					derivative = (output_mask == 0 || output_mask[t]) * learning_rate * (neuron - target_output[h][t][j]);

					if(strstr(type_layer[i], "ce")){
						if(strstr(type_layer[i], "sm")){
							// derivative = derivative;
						}
						else{
							// derivative = derivative;
						}
					}
					else
					if(strstr(type_layer[i], "mse")){
						if(strstr(type_layer[i], "ht")){
							derivative *= (1 - neuron) * (1 + neuron);
						}
						else
						if(strstr(type_layer[i], "ia")){
							// derivative *= 1;
						}
						else{
							derivative *= (1 - neuron) * neuron;
						}
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Feedforward(char option[], int layer_index, int time_index, int map_index){
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double ****neuron						= this->neuron[0][0][i][t];
	double ****neuron_patch[2]				= {this->neuron[0][1][i][t], this->neuron[0][2][i][t]};
	double ****lower_neuron					= this->neuron[0][0][i - 1][t];
	double ****previous_neuron				= this->neuron[0][0][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double ****reset_neuron					= this->neuron[1][0][i][t];
	double ****reset_neuron_patch[2]		= {this->neuron[1][1][i][t], this->neuron[1][2][i][t]};

	double ****update_neuron				= this->neuron[2][0][i][t];
	double ****update_neuron_patch[2]		= {this->neuron[2][1][i][t], this->neuron[2][2][i][t]};

	double ****block_neuron					= this->neuron[3][0][i][t];
	double ****block_neuron_patch[2]		= {this->neuron[3][1][i][t], this->neuron[3][2][i][t]};
		
	double ****forget_neuron				= this->neuron[1][0][i][t];
	double ****forget_neuron_patch[2]		= {this->neuron[1][1][i][t], this->neuron[1][2][i][t]};

	double ****input_neuron					= this->neuron[2][0][i][t];
	double ****input_neuron_patch[2]		= {this->neuron[2][1][i][t], this->neuron[2][2][i][t]};

	double ****output_neuron				= this->neuron[3][0][i][t];
	double ****output_neuron_patch[2]		= {this->neuron[3][1][i][t], this->neuron[3][2][i][t]};

	double ****block_cell_neuron			= this->neuron[4][0][i][t];
	double ****block_cell_neuron_patch[2]	= {this->neuron[4][1][i][t], this->neuron[4][2][i][t]};

	double ****cell_neuron					= this->neuron[5][0][i][t];
	double ****cell_neuron_patch			= this->neuron[5][1][i][t];
	double ****previous_cell_neuron			= this->neuron[5][1][i][(t - 1 < 0) ? (time_step):(t - 1)];

	double *forget_cell_weight	= this->cell_weight[1][i];
	double *input_cell_weight	= this->cell_weight[2][i];
	double *output_cell_weight	= this->cell_weight[3][i];

	double ****weight			= this->weight[0][i];
	double ****reset_weight		= this->weight[1][i];
	double ****update_weight	= this->weight[2][i];
		
	double ****forget_weight	= this->weight[1][i];
	double ****input_weight		= this->weight[2][i];
	double ****output_weight	= this->weight[3][i];

	if(strstr(option, "A")){
		if(type_layer[i][0] == 'C' || type_layer[i][0] == 'L'){
			if(strstr(type_layer[i], "gru")){
				Feedforward(layer_index, map_index, reset_neuron, reset_neuron_patch, lower_neuron, previous_neuron, 0, 0, 0, reset_weight);
				Feedforward(layer_index, map_index, update_neuron, update_neuron_patch, lower_neuron, previous_neuron, 0, 0, 0, update_weight);
			}
			else
			if(strstr(type_layer[i], "lstm")){
				Feedforward(layer_index, map_index, forget_neuron, forget_neuron_patch, lower_neuron, previous_neuron, previous_cell_neuron, 0, forget_cell_weight, forget_weight);
				Feedforward(layer_index, map_index, input_neuron, input_neuron_patch, lower_neuron, previous_neuron, previous_cell_neuron, 0, input_cell_weight, input_weight);
				Feedforward(layer_index, map_index, block_cell_neuron, block_cell_neuron_patch, lower_neuron, previous_neuron, 0, 0, 0, weight);
			}
			else
			if(strstr(type_layer[i], "rc")){
				Feedforward(layer_index, map_index, neuron, neuron_patch, lower_neuron, previous_neuron, 0, 0, 0, weight);
			}
			else{
				Feedforward(layer_index, map_index, neuron, 0, lower_neuron, 0, 0, 0, 0, weight);
			}
		}
		else
		if(type_layer[i][0] == 'P'){
			if(strstr(type_layer[i], "pad")){
				int margin[] = {(map_height[i] - map_height[i - 1]) / 2, (map_width[i] - map_width[i - 1]) / 2};

				for(int h = 0;h < batch_size;h++){
					double **lower_neuron	= this->neuron[0][0][i - 1][t][h][j];
					double **neuron			= this->neuron[0][0][i][t][h][j];

					for(int k = 0;k < map_height[i];k++){
						for(int l = 0;l < map_width[i];l++){
							neuron[k][l] = 0;
						}
					}
					for(int k = 0;k < map_height[i - 1];k++){
						for(int l = 0;l < map_width[i - 1];l++){
							neuron[margin[0] + k][margin[1] + l] = lower_neuron[k][l];
						}
					}
				}
			}
			else{
				int stride[] = {map_height[i - 1] / map_height[i], map_width[i - 1] / map_width[i]};

				for(int h = 0;h < batch_size;h++){
					double **lower_neuron	= this->neuron[0][0][i - 1][t][h][j];
					double **neuron			= this->neuron[0][0][i][t][h][j];

					for(int k = 0;k < map_height[i];k++){
						for(int l = 0;l < map_width[i];l++){
							if(strstr(type_layer[i], "avg")){
								double sum = 0;
						
								for(int m = 0;m < stride[0];m++){
									for(int n = 0;n < stride[1];n++){
										sum += lower_neuron[k * stride[0] + m][l * stride[1] + n];
									}
								}
								neuron[k][l] = sum / (stride[0] * stride[1]);
							}
							else
							if(strstr(type_layer[i], "max")){
								double max = -1;
						
								for(int m = 0;m < stride[0];m++){
									for(int n = 0;n < stride[1];n++){
										if(max < lower_neuron[k * stride[0] + m][l * stride[1] + n]){
											max = lower_neuron[k * stride[0] + m][l * stride[1] + n];
										}
									}
								}
								neuron[k][l] = max;
							}
						}
					}
				}
			}
		}
	}
	else
	if(strstr(option, "B")){
		if(strstr(type_layer[i], "gru")){
			Feedforward(layer_index, map_index, block_neuron, block_neuron_patch, lower_neuron, previous_neuron, 0, reset_neuron, 0, weight);

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Activate(option, 3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 3, 2, layer_index, time_index, map_index);
			}
			for(int h = 0;h < batch_size;h++){
				double mask = 1;

				if(strstr(type_layer[i], "do")){
					if(strstr(option, "train")){
						mask = dropout_mask[i][h][j];
					}
					else
					if(strstr(option, "test")){
						mask = atof(strstr(type_layer[i], "do") + 2);
					}
				}
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						block_neuron[h][j][k][l] = Tangent(block_neuron[h][j][k][l] + block_neuron_patch[0][h][j][k][l] + block_neuron_patch[1][h][j][k][l]);
						neuron[h][j][k][l]		 = (1 - update_neuron[h][j][k][l]) * previous_neuron[h][j][k][l] + update_neuron[h][j][k][l] * mask * block_neuron[h][j][k][l];
					}
				}
			}
		}
		else
		if(strstr(type_layer[i], "lstm")){
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						cell_neuron_patch[h][j][k][l] = cell_neuron[h][j][k][l];
					}
				}
			}
			Feedforward(layer_index, map_index, output_neuron, output_neuron_patch, lower_neuron, previous_neuron, cell_neuron, 0, output_cell_weight, output_weight);

			if(strstr(type_layer[i], "bn")){
				Batch_Normalization_Activate(option, 3, 1, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 3, 2, layer_index, time_index, map_index);
				Batch_Normalization_Activate(option, 5, 0, layer_index, time_index, map_index);
			}
			for(int h = 0;h < batch_size;h++){
				for(int k = 0;k < map_height[i];k++){
					for(int l = 0;l < map_width[i];l++){
						output_neuron[h][j][k][l]	= Sigmoid(output_neuron[h][j][k][l] + output_neuron_patch[0][h][j][k][l] + output_neuron_patch[1][h][j][k][l]);
						neuron[h][j][k][l]			= output_neuron[h][j][k][l] * Tangent(cell_neuron[h][j][k][l]);
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Feedforward(int layer_index, int map_index, double ****neuron, double *****neuron_patch, double ****lower_neuron, double ****previous_neuron, double ****cell_neuron, double ****reset_neuron, double *cell_weight, double ****weight){
	int i = layer_index;
	int j = map_index;

	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				neuron[h][j][k][l] = weight[j][number_maps[i - 1] + number_maps[i]][0][0];

				if(lower_neuron){
					double sum = 0;

					for(int m = 0;m < number_maps[i - 1];m++){
						for(int n = 0;n < kernel_height[i];n++){
							for(int o = 0;o < kernel_width[i];o++){
								int index[2] = {k * stride_height[i] + n, l * stride_width[i] + o};

								if(index[0] < map_height[i - 1] && index[1] < map_width[i - 1]){
									sum += lower_neuron[h][m][index[0]][index[1]] * weight[j][m][n][o];
								}
							}
						}
					}
					if(neuron_patch){
						neuron_patch[0][h][j][k][l] = sum;
					}
					else{
						neuron[h][j][k][l] += sum;
					}
				}
				if(previous_neuron){
					double sum = 0;

					for(int m = 0;m < number_maps[i];m++){
						sum += previous_neuron[h][m][k][l] * ((reset_neuron) ? (reset_neuron[h][m][k][l]):(1)) * weight[j][number_maps[i - 1] + m][0][0];
					}
					neuron_patch[1][h][j][k][l] = sum;
				}
				if(cell_neuron){
					neuron[h][j][k][l] += cell_neuron[h][j][k][l] * cell_weight[j];
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Softmax(int layer_index, int time_index){
	int i = layer_index;
	int t = time_index;

	if(strstr(type_layer[i], "sm")){
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					double max = 0;
					double sum = 0;

					double ***neuron = this->neuron[0][0][i][t][h];

					for(int j = 0;j < number_maps[i];j++){
						if(max < neuron[j][k][l]){
							max = neuron[j][k][l];
						}
					}
					for(int j = 0;j < number_maps[i];j++){
						neuron[j][k][l] = exp(neuron[j][k][l] - max);
						sum += neuron[j][k][l];
					}
					for(int j = 0;j < number_maps[i];j++){
						neuron[j][k][l] /= sum;
					}
				}
			}
		}
	}
}

void Recurrent_Neural_Networks::Batch_Normalization_Activate(char option[], int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index){
	int g = memory_type;
	int h = memory_patch_index;
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double gamma		 = this->gamma[i][j][g][h];
	double beta			 = this->beta[i][j][g][h];
	double &mean		 = this->mean[i][j][g][h][t + (test_time_index % networks_time_step)];
	double &variance	 = this->variance[i][j][g][h][t + (test_time_index % networks_time_step)];
	double &sum_mean	 = this->sum_mean[i][j][g][h][t];
	double &sum_variance = this->sum_variance[i][j][g][h][t];

	double ****neuron			= this->neuron[g][h][i][t];
	double ****neuron_batch[2]	= {this->neuron[g][2 * (h / 2) + 3][i][t], this->neuron[g][2 * (h / 2) + 4][i][t]};

	if(strstr(option, "train")){
		double sum = 0;

		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					sum += neuron[h][j][k][l];
				}
			}
		}
		sum_mean += (mean = sum / (batch_size * map_height[i] * map_width[i]));
							
		sum = 0;
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					sum += (neuron[h][j][k][l] - mean) * (neuron[h][j][k][l] - mean);
				}
			}
		}
		sum_variance += (variance = sum / (batch_size * map_height[i] * map_width[i]));
			
		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					neuron_batch[0][h][j][k][l] = (neuron[h][j][k][l] - mean) / sqrt(variance + epsilon);
					neuron_batch[1][h][j][k][l] = neuron[h][j][k][l];

					neuron[h][j][k][l] = gamma * neuron_batch[0][h][j][k][l] + beta;
				}
			}
		}
	}
	else
	if(strstr(option, "test")){
		double stdv = sqrt(variance + epsilon);

		for(int h = 0;h < batch_size;h++){
			for(int k = 0;k < map_height[i];k++){
				for(int l = 0;l < map_width[i];l++){
					neuron[h][j][k][l] = gamma / stdv * neuron[h][j][k][l] + (beta - gamma * mean / stdv);
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Batch_Normalization_Adjust_Parameter(int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index){
	int g = memory_type;
	int h = memory_patch_index;
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double sum = 0;

	double &gamma	= this->gamma_momentum[i][j][g][h];
	double &beta	= this->beta_momentum[i][j][g][h];

	double ****derivative_batch	= this->derivative[g][2 * (h / 2) + 4][i][t];
	double ****neuron_batch		= this->neuron[g][2 * (h / 2) + 3][i][t];
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				sum += derivative_batch[h][j][k][l] * neuron_batch[h][j][k][l];
			}
		}
	}
	gamma -= sum;
						
	sum = 0;
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				sum += derivative_batch[h][j][k][l];
			}
		}
	}
	beta -= sum;
}
void Recurrent_Neural_Networks::Batch_Normalization_Differentiate(int memory_type, int memory_patch_index, int layer_index, int time_index, int map_index){
	int g = memory_type;
	int h = memory_patch_index;
	int i = layer_index;
	int t = time_index;
	int j = map_index;

	double derivative_mean;
	double derivative_variance;
	double sum = 0;

	double gamma	= this->gamma[i][j][g][h];
	double beta		= this->beta[i][j][g][h];
	double mean		= this->mean[i][j][g][h][t];
	double variance	= this->variance[i][j][g][h][t];

	double ****derivative			= this->derivative[g][h][i][t];
	double ****derivative_batch[2]	= {this->derivative[g][2 * (h / 2) + 3][i][t], this->derivative[g][2 * (h / 2) + 4][i][t]};
	double ****neuron_batch[2]		= {this->neuron[g][2 * (h / 2) + 3][i][t], this->neuron[g][2 * (h / 2) + 4][i][t]};
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				derivative_batch[0][h][j][k][l] = derivative[h][j][k][l] * gamma;
				sum += derivative_batch[0][h][j][k][l] * (neuron_batch[1][h][j][k][l] - mean);
			}
		}
	}
	derivative_variance = sum * (-0.5) * pow(variance + epsilon, (double)-1.5);
				
	sum = 0;
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				sum += derivative_batch[0][h][j][k][l];
			}
		}
	}
	derivative_mean = -sum / sqrt(variance + epsilon);
		
	for(int h = 0;h < batch_size;h++){
		for(int k = 0;k < map_height[i];k++){
			for(int l = 0;l < map_width[i];l++){
				derivative_batch[1][h][j][k][l] = derivative[h][j][k][l];

				derivative[h][j][k][l] = derivative_batch[0][h][j][k][l] / sqrt(variance + epsilon) + derivative_variance * 2 * (neuron_batch[1][h][j][k][l] - mean) / (batch_size * map_height[i] * map_width[i]) + derivative_mean / (batch_size * map_height[i] * map_width[i]);
			}
		}
	}
}

void Recurrent_Neural_Networks::Gradient_Clipping(double threshold){
	double gradient = 0;

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			#pragma omp parallel for
			for(int j = 0;j < number_maps[i];j++){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							#pragma omp atomic
							gradient += (gamma_momentum[i][j][k][l] * gamma_momentum[i][j][k][l]);
							#pragma omp atomic
							gradient += (beta_momentum[i][j][k][l] * beta_momentum[i][j][k][l]);
						}
					}
				}
			}
		}
	}

	for(int h = 0;h < number_weight_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				#pragma omp parallel for
				for(int j = 0;j < number_maps[i];j++){
					#pragma omp atomic
					gradient += (cell_weight_momentum[h][i][j] * cell_weight_momentum[h][i][j]);
				}
			}
			if(Access_Weight(h, i)){
				#pragma omp parallel for
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						if(k < number_maps[i - 1]){
							for(int l = 0;l < kernel_height[i];l++){
								for(int m = 0;m < kernel_width[i];m++){
									#pragma omp atomic
									gradient += (weight_momentum[h][i][j][k][l][m] * weight_momentum[h][i][j][k][l][m]);
								}
							}
						}
					}
				}
			}
		}
	}

	if(threshold < (gradient = sqrt(gradient))){
		printf("[Gradient_Clipping], [gradient L2 norm: %lf]\n", gradient);

		gradient_factor = threshold / gradient;
	}
	else{
		gradient_factor = 1;
	}
}
void Recurrent_Neural_Networks::Refer_Memory(char option[], int time_index){
	if(time_index < 0 || time_step < time_index){
		return;
	}

	int t = time_index;

	for(int g = 0;g < number_memory_types;g++){
		for(int h = 0;h < number_memory_parts + number_memory_batches;h++){
			for(int i = 0;i < number_layers;i++){
				if(Access_Memory(g, h, i)){
					if(strstr(option, "derivative")){
						if(strstr(option, "new")){
							derivative[g][h][i][t] = new double***[batch_size];

							for(int m = 0;m < batch_size;m++){
								derivative[g][h][i][t][m] = new double**[number_maps[i]];

								for(int j = 0;j < number_maps[i];j++){
									derivative[g][h][i][t][m][j] = new double*[map_height[i]];

									for(int k = 0;k < map_height[i];k++){
										derivative[g][h][i][t][m][j][k] = new double[map_width[i]];

										if(i == number_layers - 1){
											for(int l = 0;l < map_width[i];l++){
												derivative[g][h][i][t][m][j][k][l] = 0;
											}
										}
									}
								}
							}
						}
						else
						if(strstr(option, "delete")){
							for(int m = 0;m < batch_size;m++){
								for(int j = 0;j < number_maps[i];j++){
									for(int k = 0;k < map_height[i];k++){
										delete[] derivative[g][h][i][t][m][j][k];
									}
									delete[] derivative[g][h][i][t][m][j];
								}
								delete[] derivative[g][h][i][t][m];
							}
							delete[] derivative[g][h][i][t];
						}
						else
						if(strstr(option, "zeroise")){
							for(int m = 0;m < batch_size;m++){
								for(int j = 0;j < number_maps[i];j++){
									for(int k = 0;k < map_height[i];k++){
										for(int l = 0;l < map_width[i];l++){
											derivative[g][h][i][t][m][j][k][l] = 0;
										}
									}
								}
							}
						}
					}
					else
					if(strstr(option, "neuron")){
						if(strstr(option, "zeroise")){
							for(int m = 0;m < batch_size;m++){
								for(int j = 0;j < number_maps[i];j++){
									for(int k = 0;k < map_height[i];k++){
										for(int l = 0;l < map_width[i];l++){
											neuron[g][h][i][t][m][j][k][l] = 0;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Refer_Parameter(char option[], char type_parameter_A[], char type_parameter_B[], double factor){
	int option_index;

	double ***cell_weight[2];

	double ****gamma[2];
	double ****beta[2];

	double ******weight[2];

	if(!strcmp(type_parameter_A, "parameter")){
		gamma[0]		= this->gamma;
		beta[0]			= this->beta;
		cell_weight[0]	= this->cell_weight;
		weight[0]		= this->weight;
	}
	else
	if(!strcmp(type_parameter_A, "momentum")){
		gamma[0]		= this->gamma_momentum;
		beta[0]			= this->beta_momentum;
		cell_weight[0]	= this->cell_weight_momentum;
		weight[0]		= this->weight_momentum;
	}
	else{
		fprintf(stderr, "[Refer Parameter], [%s is unknown type]\n", type_parameter_A);
	}

	if(type_parameter_B){
		if(!strcmp(type_parameter_B, "parameter")){
			gamma[1]		= this->gamma;
			beta[1]			= this->beta;
			cell_weight[1]	= this->cell_weight;
			weight[1]		= this->weight;
		}
		else
		if(!strcmp(type_parameter_B, "momentum")){
			gamma[1]		= this->gamma_momentum;
			beta[1]			= this->beta_momentum;
			cell_weight[1]	= this->cell_weight_momentum;
			weight[1]		= this->weight_momentum;
		}
		else{
			fprintf(stderr, "[Refer Parameter], [%s is unknown type]\n", type_parameter_B);
		}
	}

	if(!strcmp(option, "add")){
		option_index = 0;
	}
	else
	if(!strcmp(option, "multiply")){
		option_index = 1;
	}
	else
	if(!strcmp(option, "zeroise")){
		option_index = 2;
	}

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			#pragma omp parallel for
			for(int j = 0;j < number_maps[i];j++){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							switch(option_index){
							case 0:
								gamma[1][i][j][k][l] += gamma[0][i][j][k][l];
								beta[1][i][j][k][l]	 += beta[0][i][j][k][l];
								break;
							case 1:
								gamma[0][i][j][k][l] *= factor;
								beta[0][i][j][k][l]	 *= factor;
								break;
							case 2:
								gamma[0][i][j][k][l] = 0;
								beta[0][i][j][k][l]	 = 0;
							}
						}
					}
				}
			}
		}
	}

	for(int h = 0;h < number_weight_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				#pragma omp parallel for
				for(int j = 0;j < number_maps[i];j++){
					switch(option_index){
					case 0:
						cell_weight[1][h][i][j] += cell_weight[0][h][i][j];
						break;
					case 1:
						cell_weight[0][h][i][j] *= factor;
						break;
					case 2:
						cell_weight[0][h][i][j] = 0;
					}
				}
			}
			if(Access_Weight(h, i)){
				#pragma omp parallel for
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						for(int l = 0;l < kernel_height[i];l++){
							for(int m = 0;m < kernel_width[i];m++){
								switch(option_index){
								case 0:
									weight[1][h][i][j][k][l][m]	+= weight[0][h][i][j][k][l][m];
									break;
								case 1:
									weight[0][h][i][j][k][l][m] *= factor;
									break;
								case 2:
									weight[0][h][i][j][k][l][m] = 0;
								}
							}
						}
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Resize_Memory(int batch_size, int time_step){
	if(this->batch_size != batch_size || this->time_step != time_step){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "bn")){
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_memory_types;k++){
						if(Access_Memory(k, 0, i)){
							for(int l = 0;l < number_memory_parts;l++){
								sum_mean[i][j][k][l]	 = (double*)realloc(sum_mean[i][j][k][l], sizeof(double) * time_step);
								sum_variance[i][j][k][l] = (double*)realloc(sum_variance[i][j][k][l], sizeof(double) * time_step);
							}
						}
					}
				}
			}
		}

		for(int g = 0;g < number_memory_types;g++){
			for(int h = 0;h < number_memory_parts + number_memory_batches;h++){
				for(int i = 0;i < number_layers;i++){
					if(Access_Memory(g, h, i)){
						for(int t = 0;t < this->time_step + 1;t++){
							for(int m = 0;m < this->batch_size;m++){
								for(int j = 0;j < number_maps[i];j++){
									for(int k = 0;k < map_height[i];k++){
										delete[] derivative[g][h][i][t][m][j][k];
										delete[] neuron[g][h][i][t][m][j][k];
									}
									delete[] derivative[g][h][i][t][m][j];
									delete[] neuron[g][h][i][t][m][j];
								}
								delete[] derivative[g][h][i][t][m];
								delete[] neuron[g][h][i][t][m];
							}
							delete[] derivative[g][h][i][t];
							delete[] neuron[g][h][i][t];
						}

						derivative[g][h][i]	= (double*****)realloc(derivative[g][h][i], sizeof(double****) * (time_step + 1));
    					neuron[g][h][i]		= (double*****)realloc(neuron[g][h][i],		sizeof(double****) * (time_step + 1));

						for(int t = 0;t < time_step + 1;t++){
							derivative[g][h][i][t]	= new double***[batch_size];
							neuron[g][h][i][t]		= new double***[batch_size];

							for(int m = 0;m < batch_size;m++){
								derivative[g][h][i][t][m]	= new double**[number_maps[i]];
								neuron[g][h][i][t][m]		= new double**[number_maps[i]];

								for(int j = 0;j < number_maps[i];j++){
									derivative[g][h][i][t][m][j] = new double*[map_height[i]];
									neuron[g][h][i][t][m][j]	 = new double*[map_height[i]];

									for(int k = 0;k < map_height[i];k++){
										derivative[g][h][i][t][m][j][k]	= new double[map_width[i]];
										neuron[g][h][i][t][m][j][k]		= new double[map_width[i]];
									}
								}
							}
						}
					}
				}
			}
		}
		this->batch_size = batch_size;
		this->time_step	 = time_step;
	}
}

bool Recurrent_Neural_Networks::Access_Memory(int memory_type, int memory_patch_index, int layer_index){
	int g = memory_type;
	int h = memory_patch_index;
	int i = layer_index;

	return (g == 0 || (g < 4 && strstr(type_layer[i], "gru")) || strstr(type_layer[i], "lstm")) && (h == 0 || ((strstr(type_layer[i], "gru") || strstr(type_layer[i], "lstm") || strstr(type_layer[i], "rc")) && (h < 3 || strstr(type_layer[i], "bn"))) || (strstr(type_layer[i], "bn") && (h == 3 || h == 4)));
}
bool Recurrent_Neural_Networks::Access_Weight(int weight_type, int layer_index){
	int h = weight_type;
	int i = layer_index;

	return (h == 0 || (h < 3 && strstr(type_layer[i], "gru")) || strstr(type_layer[i], "lstm")) && (kernel_width[i] > 0);
}

double Recurrent_Neural_Networks::Tangent(double x){
	return 2 / (1 + exp(-2 * x)) - 1;
}
double Recurrent_Neural_Networks::Sigmoid(double x){
	return 1 / (1 + exp(-x));
}

Recurrent_Neural_Networks::Recurrent_Neural_Networks(char **type_layer, int number_layers, int time_step, int map_width[], int map_height[], int number_maps[]){
	this->kernel_height	= new int[number_layers];
	this->kernel_width	= new int[number_layers];
	this->map_width		= new int[number_layers];
	this->map_height	= new int[number_layers];
	this->number_layers	= number_layers;
	this->number_maps	= new int[number_layers];
	this->stride_width	= new int[number_layers];
	this->stride_height	= new int[number_layers];
	this->type_layer	= new char*[number_layers];

	batch_size				= 1;
	this->time_step			= time_step;
	networks_time_step		= time_step;
	number_memory_batches	= 4;
	number_memory_parts		= 3;
	number_memory_types		= 6;
	number_weight_types		= 4;

	for(int i = 0;i < number_layers;i++){
		this->type_layer[i] = new char[strlen(type_layer[i]) + 1];
		strcpy(this->type_layer[i], type_layer[i]);
		this->number_maps[i] = number_maps[i];
		this->map_width[i]	 = (map_width == 0) ? (1):(map_width[i]);
		this->map_height[i]	 = (map_height == 0) ? (1):(map_height[i]);

		if(strstr(type_layer[i], "ks")){
			char *kernel_size = strstr(type_layer[i], "ks");

			kernel_width[i] = atoi(kernel_size + 2);
			kernel_size = strstr(kernel_size, ",");
			kernel_height[i] = (kernel_size && atoi(kernel_size + 1) > 0) ? (atoi(kernel_size + 1)):(kernel_width[i]);
		}
		else{
			kernel_width[i]	 = (i == 0 || type_layer[i][0] == 'P') ? (0):(this->map_width[i - 1] - this->map_width[i] + 1);
			kernel_height[i] = (i == 0 || type_layer[i][0] == 'P') ? (0):(this->map_height[i - 1] - this->map_height[i] + 1);
		}

		if(strstr(type_layer[i], "st")){
			char *stride = strstr(type_layer[i], "st");

			stride_width[i] = atoi(stride + 2);
			stride = strstr(stride, ",");
			stride_height[i] = (stride && atoi(stride + 1) > 0) ? (atoi(stride + 1)):(stride_width[i]);
		}
		else{
			stride_width[i]	 = 1;
			stride_height[i] = 1;
		}
	}

	gamma			= new double***[number_layers];
	gamma_momentum	= new double***[number_layers];
	beta			= new double***[number_layers];
	beta_momentum	= new double***[number_layers];
	mean			= new double****[number_layers];
	variance		= new double****[number_layers];
	sum_mean		= new double****[number_layers];
	sum_variance	= new double****[number_layers];

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			gamma[i]			= new double**[number_maps[i]];
			gamma_momentum[i]	= new double**[number_maps[i]];
			beta[i]				= new double**[number_maps[i]];
			beta_momentum[i]	= new double**[number_maps[i]];
			mean[i]				= new double***[number_maps[i]];
			variance[i]			= new double***[number_maps[i]];
			sum_mean[i]			= new double***[number_maps[i]];
			sum_variance[i]		= new double***[number_maps[i]];

			for(int j = 0;j < number_maps[i];j++){
				gamma[i][j]				= new double*[number_memory_types];
				gamma_momentum[i][j]	= new double*[number_memory_types];
				beta[i][j]				= new double*[number_memory_types];
				beta_momentum[i][j]		= new double*[number_memory_types];
				mean[i][j]				= new double**[number_memory_types];
				variance[i][j]			= new double**[number_memory_types];
				sum_mean[i][j]			= new double**[number_memory_types];
				sum_variance[i][j]		= new double**[number_memory_types];

				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						gamma[i][j][k]			= new double[number_memory_parts];
						gamma_momentum[i][j][k]	= new double[number_memory_parts];
						beta[i][j][k]			= new double[number_memory_parts];
						beta_momentum[i][j][k]	= new double[number_memory_parts];
						mean[i][j][k]			= new double*[number_memory_parts];
						variance[i][j][k]		= new double*[number_memory_parts];
						sum_mean[i][j][k]		= new double*[number_memory_parts];
						sum_variance[i][j][k]	= new double*[number_memory_parts];

						for(int l = 0;l < number_memory_parts;l++){
							mean[i][j][k][l]		 = new double[time_step];
							variance[i][j][k][l]	 = new double[time_step];
							sum_mean[i][j][k][l]	 = new double[time_step];
							sum_variance[i][j][k][l] = new double[time_step];
						}
					}
				}
			}
		}
	}

	derivative	= new double*******[number_memory_types];
	neuron		= new double*******[number_memory_types];

	for(int g = 0;g < number_memory_types;g++){
		derivative[g]	= new double******[number_memory_parts + number_memory_batches];
		neuron[g]		= new double******[number_memory_parts + number_memory_batches];

		for(int h = 0;h < number_memory_parts + number_memory_batches;h++){
			derivative[g][h]	= new double*****[number_layers];
			neuron[g][h]		= new double*****[number_layers];

			for(int i = 0;i < number_layers;i++){
				derivative[g][h][i]	= new double****[time_step + 1];
				neuron[g][h][i]		= new double****[time_step + 1];

				if(Access_Memory(g, h, i)){
					for(int t = 0;t < time_step + 1;t++){
						derivative[g][h][i][t]	= new double***[batch_size];
						neuron[g][h][i][t]		= new double***[batch_size];

						for(int m = 0;m < batch_size;m++){
							derivative[g][h][i][t][m]	= new double**[number_maps[i]];
							neuron[g][h][i][t][m]		= new double**[number_maps[i]];

							for(int j = 0;j < number_maps[i];j++){
								derivative[g][h][i][t][m][j] = new double*[this->map_height[i]];
								neuron[g][h][i][t][m][j]	 = new double*[this->map_height[i]];

								for(int k = 0;k < this->map_height[i];k++){
									derivative[g][h][i][t][m][j][k]	= new double[this->map_width[i]];
									neuron[g][h][i][t][m][j][k]		= new double[this->map_width[i]];
								}
							}
						}
					}
				}
			}
		}
	}

	cell_weight			 = new double**[number_weight_types];
	cell_weight_momentum = new double**[number_weight_types];
	weight				 = new double*****[number_weight_types];
	weight_momentum		 = new double*****[number_weight_types];

	for(int h = 0;h < number_weight_types;h++){
		cell_weight[h]			= new double*[number_layers];
		cell_weight_momentum[h] = new double*[number_layers];
		weight[h]				= new double****[number_layers];
		weight_momentum[h]		= new double****[number_layers];

		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				cell_weight[h][i]			= new double[number_maps[i]];
				cell_weight_momentum[h][i]	= new double[number_maps[i]];
			}
			if(Access_Weight(h, i)){
				weight[h][i]			= new double***[number_maps[i]];
				weight_momentum[h][i]	= new double***[number_maps[i]];

				for(int j = 0;j < number_maps[i];j++){
					weight[h][i][j]				= new double**[number_maps[i - 1] + number_maps[i] + 1];
					weight_momentum[h][i][j]	= new double**[number_maps[i - 1] + number_maps[i] + 1];

					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						weight[h][i][j][k]			= new double*[kernel_height[i]];
						weight_momentum[h][i][j][k]	= new double*[kernel_height[i]];

						for(int l = 0;l < kernel_height[i];l++){
							weight[h][i][j][k][l]			= new double[kernel_width[i]];
							weight_momentum[h][i][j][k][l]	= new double[kernel_width[i]];
						}
					}
				}
			}
		}
	}
}
Recurrent_Neural_Networks::~Recurrent_Neural_Networks(){
	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_maps[i];j++){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							delete[] mean[i][j][k][l];
							delete[] variance[i][j][k][l];
							delete[] sum_mean[i][j][k][l];
							delete[] sum_variance[i][j][k][l];
						}
						delete[] gamma[i][j][k];
						delete[] gamma_momentum[i][j][k];
						delete[] beta[i][j][k];
						delete[] beta_momentum[i][j][k];
						delete[] mean[i][j][k];
						delete[] variance[i][j][k];
						delete[] sum_mean[i][j][k];
						delete[] sum_variance[i][j][k];
					}
				}
				delete[] gamma[i][j];
				delete[] gamma_momentum[i][j];
				delete[] beta[i][j];
				delete[] beta_momentum[i][j];
				delete[] mean[i][j];
				delete[] variance[i][j];
				delete[] sum_mean[i][j];
				delete[] sum_variance[i][j];
			}
			delete[] gamma[i];
			delete[] gamma_momentum[i];
			delete[] beta[i];
			delete[] beta_momentum[i];
			delete[] mean[i];
			delete[] variance[i];
			delete[] sum_mean[i];
			delete[] sum_variance[i];
		}
	}
	delete[] gamma;
	delete[] gamma_momentum;
	delete[] beta;
	delete[] beta_momentum;
	delete[] mean;
	delete[] variance;
	delete[] sum_mean;
	delete[] sum_variance;

	for(int g = 0;g < number_memory_types;g++){
		for(int h = 0;h < number_memory_parts + number_memory_batches;h++){
			for(int i = 0;i < number_layers;i++){
				if(Access_Memory(g, h, i)){
					for(int t = 0;t < time_step + 1;t++){
						for(int m = 0;m < batch_size;m++){
							for(int j = 0;j < number_maps[i];j++){
								for(int k = 0;k < map_height[i];k++){
									delete[] derivative[g][h][i][t][m][j][k];
									delete[] neuron[g][h][i][t][m][j][k];
								}
								delete[] derivative[g][h][i][t][m][j];
								delete[] neuron[g][h][i][t][m][j];
							}
							delete[] derivative[g][h][i][t][m];
							delete[] neuron[g][h][i][t][m];
						}
						delete[] derivative[g][h][i][t];
						delete[] neuron[g][h][i][t];
					}
				}
				delete[] derivative[g][h][i];
				delete[] neuron[g][h][i];
			}
			delete[] derivative[g][h];
			delete[] neuron[g][h];
		}
		delete[] derivative[g];
		delete[] neuron[g];
	}
	delete[] derivative;
	delete[] neuron;

	for(int h = 0;h < number_weight_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				delete[] cell_weight[h][i];
				delete[] cell_weight_momentum[h][i];
			}
			if(Access_Weight(h, i)){
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						for(int l = 0;l < kernel_height[i];l++){
							delete[] weight[h][i][j][k][l];
							delete[] weight_momentum[h][i][j][k][l];
						}
						delete[] weight[h][i][j][k];
						delete[] weight_momentum[h][i][j][k];
					}
					delete[] weight[h][i][j];
					delete[] weight_momentum[h][i][j];
				}
				delete[] weight[h][i];
				delete[] weight_momentum[h][i];
			}
		}
		delete[] cell_weight[h];
		delete[] cell_weight_momentum[h];
		delete[] weight[h];
		delete[] weight_momentum[h];
	}
	delete[] cell_weight;
	delete[] cell_weight_momentum;
	delete[] weight;
	delete[] weight_momentum;

	for(int i = 0;i < number_layers;i++){
		delete[] type_layer[i];
	}
	delete[] kernel_width;
	delete[] kernel_height;
	delete[] map_width;
	delete[] map_height;
	delete[] number_maps;
	delete[] stride_width;
	delete[] stride_height;
	delete[] type_layer;
}

void Recurrent_Neural_Networks::Initialize_Parameter(int seed, double scale, double shift){
	srand(seed);

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_maps[i];j++){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							gamma[i][j][k][l]	= 0.1;
							beta[i][j][k][l]	= 0;
						}
					}
				}
			}
		}
	}

	for(int h = 0;h < number_weight_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				for(int j = 0;j < number_maps[i];j++){
					cell_weight[h][i][j] = scale * rand() / RAND_MAX + shift;
				}
			}
			if(Access_Weight(h, i)){
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						for(int l = 0;l < kernel_height[i];l++){
							for(int m = 0;m < kernel_width[i];m++){
								weight[h][i][j][k][l][m] = scale * rand() / RAND_MAX + shift;
							}
						}
					}
				}
			}
		}
	}
}
void Recurrent_Neural_Networks::Load_Parameter(char path[]){
	FILE *file = fopen(path, "rt");

	if(file){
		fscanf(file, "%lf", &epsilon);

		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "bn")){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							for(int t = 0;t < networks_time_step;t++){
								for(int j = 0;j < number_maps[i];j++) fscanf(file, "%lf", &mean[i][j][k][l][t]);
								for(int j = 0;j < number_maps[i];j++) fscanf(file, "%lf", &variance[i][j][k][l][t]);
							}
							for(int j = 0;j < number_maps[i];j++) fscanf(file, "%lf", &gamma[i][j][k][l]);
							for(int j = 0;j < number_maps[i];j++) fscanf(file, "%lf", &beta[i][j][k][l]);
						}
					}
				}
			}
		}

		for(int h = 0;h < number_weight_types;h++){
			for(int i = 0;i < number_layers;i++){
				if(strstr(type_layer[i], "lstm")){
					for(int j = 0;j < number_maps[i];j++){
						fscanf(file, "%lf", &cell_weight[h][i][j]);
					}
				}
				if(Access_Weight(h, i)){
					for(int j = 0;j < number_maps[i];j++){
						for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
							for(int l = 0;l < kernel_height[i];l++){
								for(int m = 0;m < kernel_width[i];m++){
									fscanf(file, "%lf", &weight[h][i][j][k][l][m]);
								}
							}
						}
					}
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load_Parameter], %s not found\n", path);
	}
}
void Recurrent_Neural_Networks::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	fprintf(file, "%f\n", epsilon);

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			for(int k = 0;k < number_memory_types;k++){
				if(Access_Memory(k, 0, i)){
					for(int l = 0;l < number_memory_parts;l++){
						for(int t = 0;t < networks_time_step;t++){
							for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", mean[i][j][k][l][t]);
							for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", variance[i][j][k][l][t]);
						}
						for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", gamma[i][j][k][l]);
						for(int j = 0;j < number_maps[i];j++) fprintf(file, "%f\n", beta[i][j][k][l]);
					}
				}
			}
		}
	}

	for(int h = 0;h < number_weight_types;h++){
		for(int i = 0;i < number_layers;i++){
			if(strstr(type_layer[i], "lstm")){
				for(int j = 0;j < number_maps[i];j++){
					fprintf(file, "%f\n", cell_weight[h][i][j]);
				}
			}
			if(Access_Weight(h, i)){
				for(int j = 0;j < number_maps[i];j++){
					for(int k = 0;k < number_maps[i - 1] + number_maps[i] + 1;k++){
						for(int l = 0;l < kernel_height[i];l++){
							for(int m = 0;m < kernel_width[i];m++){
								fprintf(file, "%f\n", weight[h][i][j][k][l][m]);
							}
						}
					}
				}
			}
		}
	}
	fclose(file);
}
void Recurrent_Neural_Networks::Test(bool initialize, int time_index, double input[], double output[]){
	Resize_Memory(1, 1);
	test_time_index = time_index;

	#pragma omp parallel for
	for(int h = 0;h < number_maps[0] * map_height[0] * map_width[0];h++){
		int j = (h / (map_height[0] * map_width[0]));
		int k = (h % (map_height[0] * map_width[0])) / map_width[0];
		int l = (h % (map_height[0] * map_width[0])) % map_width[0];

		neuron[0][0][0][0][0][j][k][l] = input[h];
	}

	if(initialize){
		for(int i = 1;i < number_layers;i++){
			#pragma omp parallel for
			for(int h = 0;h < number_maps[i] * map_height[i] * map_width[i];h++){
				int j = (h / (map_height[i] * map_width[i]));
				int k = (h % (map_height[i] * map_width[i])) / map_width[i];
				int l = (h % (map_height[i] * map_width[i])) % map_width[i];

				if(strstr(type_layer[i], "gru") || strstr(type_layer[i], "rc")){
					neuron[0][0][i][1][0][j][k][l] = 0;
				}
				else
				if(strstr(type_layer[i], "lstm")){
					neuron[0][0][i][1][0][j][k][l] = 0;
					neuron[5][1][i][1][0][j][k][l] = 0;
				}
			}
		}
	}

	for(int i = 1;i < number_layers;i++){
		#pragma omp parallel for
		for(int j = 0;j < number_maps[i];j++){
			Feedforward	("test/A",	i, 0, j);
		}
		#pragma omp parallel for
		for(int j = 0;j < number_maps[i];j++){
			Activate	("test",	i, 0, j);
		}
		#pragma omp parallel for
		for(int j = 0;j < number_maps[i];j++){
			Feedforward	("test/B",	i, 0, j);
		}
		Softmax(i, 0);
	}
	for(int i = number_layers - 1, j = 0;j < number_maps[i];j++){
		output[j] = neuron[0][0][i][0][0][j][0][0];
	}

	for(int i = 1;i < number_layers;i++){
		#pragma omp parallel for
		for(int h = 0;h < number_maps[i] * map_height[i] * map_width[i];h++){
			int j = (h / (map_height[i] * map_width[i]));
			int k = (h % (map_height[i] * map_width[i])) / map_width[i];
			int l = (h % (map_height[i] * map_width[i])) % map_width[i];

			if(strstr(type_layer[i], "gru") || strstr(type_layer[i], "rc")){
				neuron[0][0][i][1][0][j][k][l] = neuron[0][0][i][0][0][j][k][l];
			}
			else
			if(strstr(type_layer[i], "lstm")){
				neuron[0][0][i][1][0][j][k][l] = neuron[0][0][i][0][0][j][k][l];
				neuron[5][1][i][1][0][j][k][l] = neuron[5][1][i][0][0][j][k][l];
			}
		}
	}
}

double Recurrent_Neural_Networks::Train(int batch_size, int number_training, int time_step, int length_data[], bool _output_mask[], double epsilon, double gradient_threshold, double learning_rate, double noise_scale_factor, double ***input, double ***target_output){
	int number_batches	= 0;
	int number_data		= 0;

	int *index = new int[number_training];

	double loss = 0;

	double **gradient = new double*[number_layers];

	double ***target_output_batch = new double**[batch_size];

	std::default_random_engine generator(0);
	std::normal_distribution<double> distribution(0.0, 1.0);

	for(int i = 0;i < number_training;i++){
		index[i] = i;
	}
	for(int i = 0;i < number_training;i++){
		int j = rand() % number_training;
		int t = index[i];

		index[i] = index[j];
		index[j] = t;

		for(int j = 0;j < length_data[i];j++){
			if(_output_mask == 0 || _output_mask[j]){
				number_data++;
			}
		}
	}

	dropout_mask = new int**[number_layers];

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "do")){
			dropout_mask[i] = new int*[batch_size];

			for(int h = 0;h < batch_size;h++){
				dropout_mask[i][h] = new int[number_maps[i]];
			}
		}
	}

	for(int h = 0;h < batch_size;h++){
		target_output_batch[h] = new double*[time_step];

		for(int t = 0;t < time_step;t++){
			target_output_batch[h][t] = new double[number_maps[number_layers - 1]];
		}
	}

	test_time_index = 0;
	Resize_Memory(batch_size, time_step);
	Refer_Memory ("zeroise derivative", time_step);

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			for(int j = 0;j < number_maps[i];j++){
				for(int k = 0;k < number_memory_types;k++){
					if(Access_Memory(k, 0, i)){
						for(int l = 0;l < number_memory_parts;l++){
							for(int t = 0;t < time_step;t++){
								sum_mean[i][j][k][l][t]		= 0;
								sum_variance[i][j][k][l][t] = 0;
							}
						}
					}
				}
			}
		}
	}
	this->epsilon = epsilon;

	for(int g = 0, h = 0;g < number_training;g++){
		if(++h == batch_size){
			int maximum_length_data = 0;

			h = 0;

			for(int i = g - (batch_size - 1);i < g;i++){
				if(maximum_length_data < length_data[i]){
					maximum_length_data = length_data[i];
				}
			}
			Refer_Memory("zeroise neuron", time_step);

			for(int s = 0;s < maximum_length_data;s += time_step){
				bool *output_mask = new bool[time_step];

				for(int t = 0;t < time_step;t++){
					output_mask[t] = false;
				}
				for(int i = g - (batch_size - 1);i <= g;i++){
					int batch_index = i - g + (batch_size - 1);

					for(int t = 0;t < time_step;t++){
						if(s + t < length_data[index[i]]){
							double *noise = new double[number_maps[0] * map_height[0] * map_width[0]];

							for(int m = 0;m < number_maps[0] * map_height[0] * map_width[0];m++){
								noise[m] = noise_scale_factor * distribution(generator);
							}
							output_mask[t] = (_output_mask == 0) ? (true):(_output_mask[t]);

							#pragma omp parallel for
							for(int m = 0;m < number_maps[0] * map_height[0] * map_width[0];m++){
								int j = (m / (map_height[0] * map_width[0]));
								int k = (m % (map_height[0] * map_width[0])) / map_width[0];
								int l = (m % (map_height[0] * map_width[0])) % map_width[0];

								neuron[0][0][0][t][batch_index][j][k][l] = input[index[i]][s + t][m] + noise[m];
							}
							for(int j = 0;j < number_maps[number_layers - 1];j++){
								target_output_batch[batch_index][t][j] = target_output[index[i]][s + t][j];
							}
							delete[] noise;
						}
						else{
							output_mask[t] |= false;

							#pragma omp parallel for
							for(int m = 0;m < number_maps[0] * map_height[0] * map_width[0];m++){
								int j = (m / (map_height[0] * map_width[0]));
								int k = (m % (map_height[0] * map_width[0])) / map_width[0];
								int l = (m % (map_height[0] * map_width[0])) % map_width[0];

								neuron[0][0][0][t][batch_index][j][k][l] = 0;
							}
							for(int j = 0;j < number_maps[number_layers - 1];j++){
								target_output_batch[batch_index][t][j] = 0;
							}
						}
					}
				}

				for(int i = 0;i < number_layers;i++){
					if(strstr(type_layer[i], "do")){
						for(int h = 0;h < batch_size;h++){
							for(int j = 0;j < number_maps[i];j++){
								dropout_mask[i][h][j] = ((double)rand() / RAND_MAX <= atof(strstr(type_layer[i], "do") + 2));
							}
						}
					}
				}

				for(int t = 0;t < time_step;t++){
					for(int i = 1;i < number_layers;i++){
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Feedforward	("train/A",	i, t, j);
						}
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Activate	("train",	i, t, j);
						}
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Feedforward	("train/B",	i, t, j);
						}
						Softmax(i, t);
					}
				}
				number_batches++;

				Refer_Parameter("zeroise", "momentum", 0, 0);

				for(int t = time_step - 1;t >= 0;t--){
					for(int i = number_layers - 1;i > 0;i--){
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Backpropagate	('A', i, t, j);
						}
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Differentiate	(i, t, j, output_mask, learning_rate, target_output_batch);
						}
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Backpropagate	('B', i, t, j);
						}
						#pragma omp parallel for
						for(int j = 0;j < number_maps[i];j++){
							Adjust_Parameter(i, t, j);
						}
					}
				}

				Gradient_Clipping(gradient_threshold);

				if(gradient_factor != 1){
					Refer_Parameter("multiply", "momentum", 0, gradient_factor);
				}
				Refer_Parameter("add", "momentum", "parameter", 0);

				for(int h = 0;h < batch_size;h++){
					for(int t = 0;t < time_step;t++){
						if(output_mask == 0 || output_mask[t]){
							for(int i = number_layers - 1, j = 0;j < number_maps[i];j++){
								if(strstr(type_layer[i], "ce")){
									loss -= target_output_batch[h][t][j] * log(neuron[0][0][i][t][h][j][0][0] + 0.000001) + (1 - target_output_batch[h][t][j]) * log(1 - neuron[0][0][i][t][h][j][0][0] + 0.000001);
								}
								if(strstr(type_layer[i], "mse")){
									loss += 0.5 * (neuron[0][0][i][t][h][j][0][0] - target_output_batch[h][t][j]) * (neuron[0][0][i][t][h][j][0][0] - target_output_batch[h][t][j]);
								}
							}
						}
					}
				}

				for(int h = 0;h < batch_size;h++){
					for(int i = 0;i < number_layers;i++){
						#pragma omp parallel for
						for(int m = 0;m < number_maps[i] * map_height[i] * map_width[i];m++){
							int j = (m / (map_height[i] * map_width[i]));
							int k = (m % (map_height[i] * map_width[i])) / map_width[i];
							int l = (m % (map_height[i] * map_width[i])) % map_width[i];

							if(strstr(type_layer[i], "gru") || strstr(type_layer[i], "rc")){
								neuron[0][0][i][time_step][0][j][k][l] = neuron[0][0][i][time_step - 1][0][j][k][l];
							}
							else
							if(strstr(type_layer[i], "lstm")){
								neuron[0][0][i][time_step][0][j][k][l] = neuron[0][0][i][time_step - 1][0][j][k][l];
								neuron[5][1][i][time_step][0][j][k][l] = neuron[5][1][i][time_step - 1][0][j][k][l];
							}
						}
					}
				}
				delete[] output_mask;
			}
		}
	}

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "bn")){
			#pragma omp parallel for
			for(int h = 0;h < number_maps[i] * number_memory_types * number_memory_parts;h++){
				int j = (h / (number_memory_types * number_memory_parts));
				int k = (h % (number_memory_types * number_memory_parts)) / number_memory_parts;
				int l = (h % (number_memory_types * number_memory_parts)) % number_memory_parts;

				if(Access_Memory(k, 0, i)){
					for(int t = 0;t < time_step;t++){
						mean[i][j][k][l][t]		= sum_mean[i][j][k][l][t] / number_batches;
						variance[i][j][k][l][t]	= (double)batch_size / (batch_size - 1) * sum_variance[i][j][k][l][t] / number_batches;
					}
				}
			}
		}
	}

	for(int i = 0;i < number_layers;i++){
		if(strstr(type_layer[i], "do")){
			for(int h = 0;h < batch_size;h++){
				delete[] dropout_mask[i][h];
			}
			delete[] dropout_mask[i];
		}
	}
	delete[] dropout_mask;

	for(int h = 0;h < batch_size;h++){
		for(int t = 0;t < time_step;t++){
			delete[] target_output_batch[h][t];
		}
		delete[] target_output_batch[h];
	}
	delete[] index;
	delete[] target_output_batch;

	return loss / number_data;
}
