#include <fstream>
#include <iostream>
#include <sstream>

#include "Speech_Processing.h"

using namespace std;

void Speech_Processing::Discrete_Cosine_Transform(int direction, int length, double X[]) {
	double *x = new double[length];

	for (int i = 0; i < length; i++) {
		x[i] = X[i];
	}
	for (int k = 0; k < length; k++) {
		double sum = 0;

		if (direction == 1) {
			for (int n = 0; n < length; n++) {
				sum += ((k == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(3.14159265358979323846 * (n + 0.5) * k / length);
			}
		}
		else if (direction == -1) {
			for (int n = 0; n < length; n++) {
				sum += ((n == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(3.14159265358979323846 * n * (k + 0.5) / length);
			}
		}
		X[k] = sum * sqrt(2.0 / length);
	}
	delete[] x;
}
void Speech_Processing::DCT(int direction, int length, double X[]) {
	if (direction == 1 || direction == -1) {
		Discrete_Cosine_Transform(direction, length, X);
		return;
	}
	fprintf(stderr, "[DCT], [direction = {-1 (inversed transform), 1 (forward transform)}\n");
}
void Speech_Processing::Fast_Fourier_Transform(int direction, int length, double Xr[], double Xi[]) {
	int log_length = (int)log2((double)length);

	for (int i = 0, j = 0; i < length; i++, j = 0) {
		for (int k = 0; k < log_length; k++) {
			j = (j << 1) | (1 & (i >> k));
		}
		if (j < i) {
			double t;

			t = Xr[i];
			Xr[i] = Xr[j];
			Xr[j] = t;

			t = Xi[i];
			Xi[i] = Xi[j];
			Xi[j] = t;
		}
	}
	for (int i = 0; i < log_length; i++) {
		int L = (int)pow(2.0, i);

		for (int j = 0; j < length - 1; j += 2 * L) {
			for (int k = 0; k < L; k++) {
				double argument = direction * -3.14159265358979323846 * k / L;

				double xr = Xr[j + k + L] * cos(argument) - Xi[j + k + L] * sin(argument);
				double xi = Xr[j + k + L] * sin(argument) + Xi[j + k + L] * cos(argument);

				Xr[j + k + L] = Xr[j + k] - xr;
				Xi[j + k + L] = Xi[j + k] - xi;
				Xr[j + k] = Xr[j + k] + xr;
				Xi[j + k] = Xi[j + k] + xi;
			}
		}
	}
	if (direction == -1) {
		for (int k = 0; k < length; k++) {
			Xr[k] /= length;
			Xi[k] /= length;
		}
	}
}
void Speech_Processing::FFT(int direction, int length, double Xr[], double Xi[]) {
	int log_length = (int)log2((double)length);

	if (direction != 1 && direction != -1) {
		fprintf(stderr, "[FFT], [direction = {-1 (inversed transform), 1 (forward transform)}\n");
		return;
	}
	if (1 << log_length != length) {
		fprintf(stderr, "[FFT], [length must be a power of 2]\n");
		return;
	}
	Fast_Fourier_Transform(direction, length, Xr, Xi);
}

float* Speech_Processing::Calculate_MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, float frame[]) {
	float *MFCC = new float[number_coefficients];

	double max_frequency = sample_rate / 2;
	double min_frequency = 0;
	double max_Mels_frequency = Mel_Scale(1, max_frequency);
	double min_Mels_frequency = Mel_Scale(1, min_frequency);
	double interval = (max_Mels_frequency - min_Mels_frequency) / (number_filterbanks + 1);

	double *filterbank = new double[number_filterbanks];
	double *Xr = new double[length_DFT];
	double *Xi = new double[length_DFT];

	for (int i = 0; i < number_filterbanks; i++) {
		filterbank[i] = 0;
	}
	for (int i = 0; i < length_DFT; i++) {
		Xr[i] = (i < length_frame) ? (frame[i]) : (0);
		Xi[i] = 0;
	}
	FFT(1, length_DFT, Xr, Xi);

	for (int i = 0; i < length_DFT / 2 + 1; i++) {
		double frequency = (max_frequency - min_frequency) * i / (length_DFT / 2);
		double Mel_frequency = Mel_Scale(1, frequency);
		double power = (Xr[i] * Xr[i] + Xi[i] * Xi[i]) / length_frame;

		for (int j = 0; j < number_filterbanks; j++) {
			double frequency_boundary[] = { min_Mels_frequency + interval * (j + 0), min_Mels_frequency + interval * (j + 1), min_Mels_frequency + interval * (j + 2) };

			if (frequency_boundary[0] <= Mel_frequency && Mel_frequency <= frequency_boundary[1]) {
				double lower_frequency = Mel_Scale(-1, frequency_boundary[0]);
				double upper_frequency = Mel_Scale(-1, frequency_boundary[1]);

				filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
			}
			else if (frequency_boundary[1] <= Mel_frequency && Mel_frequency <= frequency_boundary[2]) {
				double lower_frequency = Mel_Scale(-1, frequency_boundary[1]);
				double upper_frequency = Mel_Scale(-1, frequency_boundary[2]);

				filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
			}
		}
	}

	for (int i = 0; i < number_filterbanks; i++) {
		filterbank[i] = log(filterbank[i]);
	}
	DCT(1, number_filterbanks, filterbank);

	for (int i = 0; i < number_coefficients; i++) {
		MFCC[i] = filterbank[i];
	}
	delete[] filterbank;
	delete[] Xr;
	delete[] Xi;

	return MFCC;
}

double Speech_Processing::Mel_Scale(int direction, double x) {
	switch (direction) {
	case -1:
		return 700 * (exp(x / 1125.0) - 1);
	case 1:
		return 1125 * log(1 + x / 700.0);
	}
	fprintf(stderr, "[Mel_Scale], direction = {-1 (inversed transform), 1 (forward transform)}\n");
	return 0;
}

double* Speech_Processing::Calculate_MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, double frame[]) {
	double max_Mels_frequency = Mel_Scale(1, sample_rate / 2);
	double min_Mels_frequency = Mel_Scale(1, 300);
	double interval = (max_Mels_frequency - min_Mels_frequency) / (number_filterbanks + 1);

	double *filterbank = new double[number_filterbanks];
	double *MFCC = new double[number_coefficients];
	double *Xr = new double[length_DFT];
	double *Xi = new double[length_DFT];

	for (int i = 0; i < number_filterbanks; i++) {
		filterbank[i] = 0;
	}
	for (int i = 0; i < length_DFT; i++) {
		Xr[i] = (i < length_frame) ? (frame[i]) : (0);
		Xi[i] = 0;
	}
	FFT(1, length_DFT, Xr, Xi);

	for (int i = 0; i < length_DFT / 2 + 1; i++) {
		double frequency = (sample_rate / 2) * i / (length_DFT / 2);
		double Mel_frequency = Mel_Scale(1, frequency);
		double power = (Xr[i] * Xr[i] + Xi[i] * Xi[i]) / length_frame;

		for (int j = 0; j < number_filterbanks; j++) {
			double frequency_boundary[] = { min_Mels_frequency + interval * (j + 0), min_Mels_frequency + interval * (j + 1), min_Mels_frequency + interval * (j + 2) };

			if (frequency_boundary[0] <= Mel_frequency && Mel_frequency <= frequency_boundary[1]) {
				double lower_frequency = Mel_Scale(-1, frequency_boundary[0]);
				double upper_frequency = Mel_Scale(-1, frequency_boundary[1]);

				filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
			}
			else
				if (frequency_boundary[1] <= Mel_frequency && Mel_frequency <= frequency_boundary[2]) {
					double lower_frequency = Mel_Scale(-1, frequency_boundary[1]);
					double upper_frequency = Mel_Scale(-1, frequency_boundary[2]);

					filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
				}
		}
	}

	for (int i = 0; i < number_filterbanks; i++) {
		filterbank[i] = log(filterbank[i]);
	}
	DCT(1, number_filterbanks, filterbank);

	for (int i = 0; i < number_coefficients; i++) {
		MFCC[i] = filterbank[i];
	}

	delete[] filterbank;
	delete[] Xr;
	delete[] Xi;

	return MFCC;
}

Speech_Processing::Speech_Processing() {
	mean = nullptr;
	stdv = nullptr;
}
Speech_Processing::~Speech_Processing() {
	if (mean) delete[] mean;
	if (stdv) delete[] stdv;
}

void Speech_Processing::Calculate_MFCC_Delta(vector<float*> &MFCC, int number_coefficients, bool leftmost, bool rightmost) {
	for (int i = 0; i < MFCC.size(); i++) {
		int prev = (i == 0 && leftmost) ? (0) : (i - 1);
		int next = (i == MFCC.size() - 1 && rightmost) ? (MFCC.size() - 1) : (i + 1);

		MFCC[i] = (float*)realloc(MFCC[i], sizeof(float) * 2 * number_coefficients);

		for (int j = 0; j < number_coefficients; j++) {
			MFCC[i][number_coefficients + j] = (MFCC[next][j] - MFCC[prev][j]) / 2;
		}
	}
}
void Speech_Processing::Calculate_MFCC_Delta_Delta(vector<float*> &MFCC, int number_coefficients, bool leftmost, bool rightmost) {
	for (int i = 0; i < MFCC.size(); i++) {
		int prev = (i == 0 && leftmost) ? (0) : (i - 1);
		int next = (i == MFCC.size() - 1 && rightmost) ? (MFCC.size() - 1) : (i + 1);

		MFCC[i] = (float*)realloc(MFCC[i], sizeof(float) * 3 * number_coefficients);

		for (int j = number_coefficients; j < 2 * number_coefficients; j++) {
			MFCC[i][number_coefficients + j] = (MFCC[next][j] - MFCC[prev][j]) / 2;
		}
	}
}
void Speech_Processing::Calculate_MFCC_Delta(vector<double*> &MFCC, int number_coefficients, bool leftmost, bool rightmost) {
	for (int i = 0; i < MFCC.size(); i++) {
		int prev = (i == 0 && leftmost) ? (0) : (i - 1);
		int next = (i == MFCC.size() - 1 && rightmost) ? (MFCC.size() - 1) : (i + 1);

		MFCC[i] = (double*)realloc(MFCC[i], sizeof(double) * 2 * number_coefficients);

		for (int j = 0; j < number_coefficients; j++) {
			MFCC[i][number_coefficients + j] = (MFCC[next][j] - MFCC[prev][j]) / 2;
		}
	}
}
void Speech_Processing::Calculate_MFCC_Delta_Delta(vector<double*> &MFCC, int number_coefficients, bool leftmost, bool rightmost) {
	for (int i = 0; i < MFCC.size(); i++) {
		int prev = (i == 0 && leftmost) ? (0) : (i - 1);
		int next = (i == MFCC.size() - 1 && rightmost) ? (MFCC.size() - 1) : (i + 1);

		MFCC[i] = (double*)realloc(MFCC[i], sizeof(double) * 3 * number_coefficients);

		for (int j = number_coefficients; j < 2 * number_coefficients; j++) {
			MFCC[i][number_coefficients + j] = (MFCC[next][j] - MFCC[prev][j]) / 2;
		}
	}
}
void Speech_Processing::Load_Parameter(string path) {
	ifstream file(path);

	if (file.is_open()) {
		int number_coefficients;

		if (mean) delete[] mean;
		if (stdv) delete[] stdv;

		file >> number_coefficients;
		mean = new double[number_coefficients];
		stdv = new double[number_coefficients];

		for (int i = 0; i < number_coefficients; i++) {
			file >> mean[i];
		}
		for (int i = 0; i < number_coefficients; i++) {
			file >> stdv[i];
		}
		file.close();
	}
	else {
		cerr << "[Load_Parameter], " + path + " not found" << endl;
	}
}
void Speech_Processing::Save_Parameter(string path) {
	if (mean && stdv) {
		ofstream file(path);

		file << number_coefficients << endl;

		for (int i = 0; i < number_coefficients; i++) {
			file << mean[i] << endl;
		}
		for (int i = 0; i < number_coefficients; i++) {
			file << stdv[i] << endl;
		}
		file.close();
	}
	else {
		cerr << "[Save_Parameter], parameter(mean, stdv) does not exist" << endl;
	}
}
void Speech_Processing::Normalize(float _MFCC[], int _length_MFCC, int number_coefficients, bool calculate_parameter) {
	int *length_MFCC = &_length_MFCC;

	float **MFCC = &_MFCC;

	Normalize(MFCC, 1, length_MFCC, number_coefficients, calculate_parameter);
}
void Speech_Processing::Normalize(double _MFCC[], int _length_MFCC, int number_coefficients, bool calculate_parameter) {
	int *length_MFCC = &_length_MFCC;

	double **MFCC = &_MFCC;

	Normalize(MFCC, 1, length_MFCC, number_coefficients, calculate_parameter);
}
void Speech_Processing::Normalize(float **MFCC, int number_MFCCs, int length_MFCC[], int number_coefficients, bool calculate_parameter) {
	if (calculate_parameter) {
		if (mean) delete[] mean;
		if (stdv) delete[] stdv;
		mean = new double[number_coefficients];
		stdv = new double[number_coefficients];

		for (int k = 0; k < number_coefficients; k++) {
			int sum_length = 0;

			mean[k] = 0;
			stdv[k] = 0;

			for (int i = 0; i < number_MFCCs; i++) {
				for (int j = 0; j < length_MFCC[i]; j++) {
					mean[k] += MFCC[i][j * number_coefficients + k];
				}
				sum_length += length_MFCC[i];
			}
			mean[k] /= sum_length;

			for (int i = 0; i < number_MFCCs; i++) {
				for (int j = 0; j < length_MFCC[i]; j++) {
					stdv[k] += (MFCC[i][j * number_coefficients + k] - mean[k]) * (MFCC[i][j * number_coefficients + k] - mean[k]);
				}
			}
			stdv[k] = sqrt(stdv[k] / sum_length);
		}
		this->number_coefficients = number_coefficients;
	}

	for (int k = 0; k < number_coefficients; k++) {
		for (int i = 0; i < number_MFCCs; i++) {
			for (int j = 0; j < length_MFCC[i]; j++) {
				MFCC[i][j * number_coefficients + k] = (MFCC[i][j * number_coefficients + k] - mean[k]) / stdv[k];
			}
		}
	}
}
void Speech_Processing::Normalize(double **MFCC, int number_MFCCs, int length_MFCC[], int number_coefficients, bool calculate_parameter) {
	if (calculate_parameter) {
		if (mean) delete[] mean;
		if (stdv) delete[] stdv;
		mean = new double[number_coefficients];
		stdv = new double[number_coefficients];

		for (int k = 0; k < number_coefficients; k++) {
			int sum_length = 0;

			mean[k] = 0;
			stdv[k] = 0;

			for (int i = 0; i < number_MFCCs; i++) {
				for (int j = 0; j < length_MFCC[i]; j++) {
					mean[k] += MFCC[i][j * number_coefficients + k];
				}
				sum_length += length_MFCC[i];
			}
			mean[k] /= sum_length;

			for (int i = 0; i < number_MFCCs; i++) {
				for (int j = 0; j < length_MFCC[i]; j++) {
					stdv[k] += (MFCC[i][j * number_coefficients + k] - mean[k]) * (MFCC[i][j * number_coefficients + k] - mean[k]);
				}
			}
			stdv[k] = sqrt(stdv[k] / sum_length);
		}
		this->number_coefficients = number_coefficients;
	}

	for (int k = 0; k < number_coefficients; k++) {
		for (int i = 0; i < number_MFCCs; i++) {
			for (int j = 0; j < length_MFCC[i]; j++) {
				MFCC[i][j * number_coefficients + k] = (MFCC[i][j * number_coefficients + k] - mean[k]) / stdv[k];
			}
		}
	}
}

double Speech_Processing::Normalized_Edit_Distance(vector<string> &hypothesis, vector<string> &reference) {
	int **m = new int*[reference.size() + 1];

	double error_rate;

	for (int i = 0; i < reference.size() + 1; i++) {
		m[i] = new int[hypothesis.size() + 1];

		for (int j = 0; j < hypothesis.size() + 1; j++) {
			if (i == 0) {
				m[0][j] = j;
			}
			else if (j == 0) {
				m[i][0] = i;
			}
		}
	}

	for (int i = 1; i <= reference.size(); i++) {
		for (int j = 1; j <= hypothesis.size(); j++) {
			if (reference[i - 1] == hypothesis[j - 1]) {
				m[i][j] = m[i - 1][j - 1];
			}
			else {
				int deletion = m[i - 1][j] + 1;
				int insertion = m[i][j - 1] + 1;
				int substitution = m[i - 1][j - 1] + 1;

				m[i][j] = (substitution < insertion) ? ((substitution < deletion) ? (substitution) : (deletion)) : (insertion);
			}
		}
	}
	error_rate = static_cast<double>(m[reference.size()][hypothesis.size()]) / reference.size();

	for (int i = 0; i < reference.size() + 1; i++) {
		delete[] m[i];
	}
	delete[] m;

	return error_rate;
}

vector<float*> Speech_Processing::Calculate_MFCC(float data[], int length_data, int length_frame, int length_stride, int number_coefficients, int number_filterbanks, int sample_rate) {
	int length_DFT = 1 << (int)ceil(log2((double)length_frame));

	vector<float*> MFCC;

	for (int i = 0; i <= length_data - length_frame; i += length_stride) {
		float *frame = new float[length_frame];

		// pre-emphasis
		for (int j = 0; j < length_frame; j++) {
			if (i + j < length_data) {
				frame[j] = data[i + j] - 0.95 * ((i + j) ? (data[i + j - 1]) : (0));
			}
			else {
				frame[j] = 0;
			}
		}

		// windowing
		for (int j = 0; j < length_frame; j++) {
			frame[j] *= 0.54 - 0.46 * cos(2 * 3.14159265358979323846 * j / (length_frame - 1));
		}
		MFCC.push_back(Calculate_MFCC(length_frame, length_DFT, number_coefficients, number_filterbanks, sample_rate, frame));

		delete[] frame;
	}
	return MFCC;
}
vector<double*> Speech_Processing::Calculate_MFCC(double data[], int length_data, int length_frame, int length_stride, int number_coefficients, int number_filterbanks, int sample_rate) {
	int length_DFT = 1 << (int)ceil(log2((double)length_frame));

	vector<double*> MFCC;

	for (int i = 0; i <= length_data - length_frame; i += length_stride) {
		double *frame = new double[length_frame];

		// pre-emphasis
		for (int j = 0; j < length_frame; j++) {
			if (i + j < length_data) {
				frame[j] = data[i + j] - 0.95 * ((i + j) ? (data[i + j - 1]) : (0));
			}
			else {
				frame[j] = 0;
			}
		}

		// windowing
		for (int j = 0; j < length_frame; j++) {
			frame[j] *= 0.54 - 0.46 * cos(2 * 3.14159265358979323846 * j / (length_frame - 1));
		}

		MFCC.push_back(Calculate_MFCC(length_frame, length_DFT, number_coefficients, number_filterbanks, sample_rate, frame));

		delete[] frame;
	}
	return MFCC;
}