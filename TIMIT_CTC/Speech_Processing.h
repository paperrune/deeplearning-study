#ifndef SPEECH_PROCESSING_H
#define SPEECH_PROCESSING_H

#include <vector>

using namespace std;

class Speech_Processing {
private:
	int number_coefficients;

	double *mean;
	double *stdv;

	void Discrete_Cosine_Transform(int direction, int length, double X[]);
	void DCT(int direction, int length, double X[]);
	void Fast_Fourier_Transform(int direction, int length, double Xr[], double Xi[]);
	void FFT(int direction, int length, double Xr[], double Xi[]);

	float* Calculate_MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, float frame[]);

	double Mel_Scale(int direction, double x);

	double* Calculate_MFCC(int length_frame, int length_DFT, int number_coefficients, int number_filterbanks, int sample_rate, double frame[]);
public:
	Speech_Processing();
	~Speech_Processing();

	void Calculate_MFCC_Delta(vector<float*> &MFCC, int number_coefficients, bool leftmost = false, bool rightmost = false);
	void Calculate_MFCC_Delta_Delta(vector<float*> &MFCC, int number_coefficients, bool leftmost = false, bool rightmost = false);
	void Calculate_MFCC_Delta(vector<double*> &MFCC, int number_coefficients, bool leftmost = false, bool rightmost = false);
	void Calculate_MFCC_Delta_Delta(vector<double*> &MFCC, int number_coefficients, bool leftmost = false, bool rightmost = false);
	void Load_Parameter(string path);
	void Save_Parameter(string path);
	void Normalize(float MFCC[], int length_MFCC, int number_coefficients, bool calculate_parameter = false);
	void Normalize(double MFCC[], int length_MFCC, int number_coefficients, bool calculate_parameter = false);
	void Normalize(float **MFCC, int number_MFCCs, int length_MFCC[], int number_coefficients, bool calculate_parameter = false);
	void Normalize(double **MFCC, int number_MFCCs, int length_MFCC[], int number_coefficients, bool calculate_parameter = false);
	
	double Normalized_Edit_Distance(vector<string> &hypothesis, vector<string> &reference);

	vector<float*> Calculate_MFCC(float data[], int length_data, int length_frame, int length_stride, int number_coefficients, int number_filterbanks, int sample_rate);
	vector<double*> Calculate_MFCC(double data[], int length_data, int length_frame, int length_stride, int number_coefficients, int number_filterbanks, int sample_rate);
};

#endif