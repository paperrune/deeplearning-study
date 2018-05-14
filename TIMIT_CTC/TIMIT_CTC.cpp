#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../Neural_Networks.h"
#include "Speech_Processing.h"
#include "Wav.h"

void Read_Label(string path, int number_events, vector<string> **event_label) {
	ifstream file(path);

	if (file.is_open()) {
		for (int h = 0; h < number_events; h++) {
			file >> path;
			path += ".PHN";

			ifstream label_file(path);

			string label_sequence;

			while (!label_file.eof()) {
				getline(label_file, label_sequence);
				label_sequence.erase(find_if(label_sequence.rbegin(), label_sequence.rend(), not1(std::ptr_fun<int, int>(isspace))).base(), label_sequence.end());

				istringstream buffer(label_sequence);

				for (string s; getline(buffer, s, ' ');) {
					if (!isdigit(*s.c_str())) {
						(*event_label)[h].push_back(s);
					}
				}
			}
		}
		file.close();
	}
	else {
		cerr << "[Read_Label], " + path + " not found\n" << endl;
	}
}

int main() {
	string label[] = { "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "jh", "ch", "b", "d", "g", "p", "t", "k", "dx", "s", "sh", "z", "zh", "f", "th", "v", "dh", "m", "n", "ng", "em", "nx", "en", "eng", "l", "r", "w", "y", "hh", "hv", "el", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "q", "pau", "epi", "h#", "" };

	int batch_size = 64;
	int dimension_event = 39;
	int number_training = 3696;
	int number_test = 192;
	int time_step = 0;

	int *length_event = new int[number_training + number_test];

	float **_event = new float*[number_training + number_test];

	vector<string> *event_label = new vector<string>[number_training + number_test];

	Speech_Processing SP;

	// Create event_label from pronunciation path
	{
		vector<string> *p = &event_label[number_training];

		Read_Label("training path.txt", number_training, &event_label);
		Read_Label("core test path.txt", number_test, &p);
	}

	// Extract MFCC from wav files in path.txt
	{
		int length_frame = 400;
		int length_stride = 160;
		int number_coefficients = 13;
		int number_filterbanks = 26;
		int nSamplesPerSec;

		ifstream file("training path.txt");

		for (int h = 0; h < number_training + number_test; h++) {
			float *buffer;

			string path;

			vector<float*> MFCC;

			Wav wav;

			if (h == number_training) {
				file.close();
				file.open("core test path.txt");
			}

			file >> path;

			system(("sph2pipe -f wav " + path + ".WAV " + path).c_str());

			wav.Load(path.c_str());
			wav.WavToBuffer();
			wav.Get_Properties(0, &nSamplesPerSec, 0);

			buffer = new float[wav.length_buffer];

			for (int i = 0; i < wav.length_buffer; i++) {
				buffer[i] = wav.Get_Buffer(i);
			}
			MFCC = SP.Calculate_MFCC(buffer, wav.length_buffer, length_frame, length_stride, number_coefficients, number_filterbanks, nSamplesPerSec);
			SP.Calculate_MFCC_Delta(MFCC, number_coefficients, true, true);
			SP.Calculate_MFCC_Delta_Delta(MFCC, number_coefficients, true, true);

			_event[h] = new float[(length_event[h] = MFCC.size()) * dimension_event];

			for (int t = 0; t < length_event[h]; t++) {
				memcpy(&_event[h][t * dimension_event], MFCC[t], sizeof(float) * dimension_event);
				delete[] MFCC[t];
			}
			delete[] buffer;

			cout << h + 1 << " / " << number_training + number_test << ", " << path << endl;
		}
		SP.Normalize(_event, number_training, length_event, 3 * number_coefficients, true);
		SP.Normalize(&_event[number_training], number_test, &length_event[number_training], 3 * number_coefficients);

		for (int h = 0; h < number_training + number_test; h++) {
			if (time_step < length_event[h]) {
				time_step = length_event[h];
			}
		}
		file.close();
	}

	// Train Neural Networks
	{
		int number_iterations = 100;

		double epsilon = 0.001;
		double gradient_threshold = 0.1;
		double learning_rate = 0.001;
		double noise = 0.6;

		vector<int> number_nodes = { dimension_event, sizeof(label) / sizeof(label[0]) };

		vector<Layer*> layer;

		// train from scratch **************************
		Neural_Networks NN = Neural_Networks(time_step);

		layer.push_back(NN.Add(new Layer("TIMIT", number_nodes.front())));
		layer.push_back(NN.Add(new Layer("LSTM", 100)));
		layer.push_back(NN.Add(new Layer("LSTM,backward", 100)));
		layer.push_back(NN.Add(new Layer("CTC,softmax", number_nodes.back())));

		layer[1]->Connect(layer[0], "W");
		layer[1]->Connect(layer[1], "W,recurrent");
		layer[2]->Connect(layer[0], "W");
		layer[2]->Connect(layer[2], "W,recurrent");
		layer[3]->Connect(layer[1], "W");
		layer[3]->Connect(layer[2], "W");

		srand(0); NN.Initialize(0.1);
		NN.Set_CTC_Loss(number_nodes.back(), label);
		// *********************************************

		// or load pretrained model
		// Neural_Networks NN = Neural_Networks("TIMIT_CTC.txt");

		NN.Set_Optimizer(new Optimizer("momentum", 0.9));
		NN.Set_Gradient_Threshold(gradient_threshold);

		for (int g = 0, time = clock(); g < number_iterations; g++) {
			if (g == 50) learning_rate *= 0.1;

			double log_likelihood = NN.Train(batch_size, number_training, length_event, _event, event_label, learning_rate, epsilon, noise);
			double LER[2] = { 0, };

			float **output = new float*[batch_size];

			for (int h = 0; h < batch_size; h++) {
				output[h] = new float[NN.time_step * number_nodes.back()];
			}
			for (int i = 0, h = 0; i < number_training + number_test; i++) {
				if (++h == batch_size || i == number_training + number_test - 1) {
					NN.Test(h, &_event[i - h + 1], output, &length_event[i - h + 1]);

					for (int g = 0; g < h; g++) {
						vector<string> hypothesis;

						NN.Decode(length_event[i - h + g + 1], output[g], hypothesis, true);

						/*for (int j = 0; j < hypothesis.size(); j++) {
							cout << hypothesis[j] << " ";
						}
						cout << endl;*/

						LER[(i - h + g + 1 < number_training) ? (0) : (1)] += SP.Normalized_Edit_Distance(hypothesis, event_label[i - h + g + 1]);
					}
					h = 0;
				}
			}
			printf(".");  NN.Save("NN.txt");
			printf("LER: %.2lf, %.2lf	L: %lf  step %d  %.2lf sec\n", 100 * LER[0] / number_training, (number_test == 0) ? (0) : (100 * LER[1] / number_test), log_likelihood, g + 1, (double)(clock() - time) / CLOCKS_PER_SEC);

			for (int h = 0; h < batch_size; h++) {
				delete[] output[h];
			}
			delete[] output;
		}
	}

	for (int h = 0; h < number_training + number_test; h++) {
		delete[] _event[h];
	}
	delete[] _event;
	delete[] event_label;
	delete[] length_event;

	return 0;
}
