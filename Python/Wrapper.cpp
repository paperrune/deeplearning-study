#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <omp.h>

#include "../Neural_Networks.h"

class Neural_Networks_Wrapper {
private:
	vector<Layer*> layer;

	Neural_Networks *NN;
public:
	Neural_Networks_Wrapper() {
		NN = new Neural_Networks();
	}
	Neural_Networks_Wrapper(int time_step) {
		NN = new Neural_Networks(time_step);
	}
	Neural_Networks_Wrapper(string path) {
		NN = new Neural_Networks(path);
	}
	~Neural_Networks_Wrapper() {
		delete NN;
	}

	void Add_Layer(string properties, int number_maps, int map_width = 1, int map_height = 1, int map_depth = 1, int coordinate_y = -1) {
		layer.push_back(NN->Add(new Layer(properties, number_maps, map_width, map_height, map_depth), coordinate_y));
	}
	void Connect(int layer_index, int parent_layer_index, string properties) {
		layer[layer_index]->Connect(layer[parent_layer_index], properties);
	}
	void Initialize(double scale, double gamma = 1) {
		NN->Initialize(scale, gamma);
	}
	void Initialize(int seed, double scale, double gamma = 1) {
		srand(seed);
		NN->Initialize(scale, gamma);
	}
	void Save(string path) {
		NN->Save(path);
	}
	void Set_Number_Threads(int number_threads) {
		omp_set_num_threads(number_threads);
	}
	void Set_Time_Mask(int layer_index, boost::python::object time_mask) {
		Py_buffer time_mask_buffer;

		if (PyObject_GetBuffer(time_mask.ptr(), &time_mask_buffer, PyBUF_SIMPLE) != -1) {
			bool *time_mask = new bool[(layer[layer_index]->time_step) ? (layer[layer_index]->time_step) : (NN->time_step)];

			memcpy(time_mask, static_cast<bool*>(time_mask_buffer.buf), (layer[layer_index]->time_step) ? (layer[layer_index]->time_step) : (NN->time_step));
			layer[layer_index]->Set_Time_Mask(time_mask);

			PyBuffer_Release(&time_mask_buffer);
		}
	}
	void Test(boost::python::object input, boost::python::object output, int _length_data = 0) {
		boost::python::list length_data;

		if (_length_data) {
			length_data.append(_length_data);
		}
		Test(1, input, output, length_data);
	}
	void Test(int batch_size, boost::python::object input, boost::python::object output) {
		boost::python::list length_data;

		Test(batch_size, input, output, length_data);
	}
	void Test(int batch_size, boost::python::object input, boost::python::object output, boost::python::list &_length_data) {
		int *length_data = (len(_length_data)) ? (new int[batch_size]) : (nullptr);

		Py_buffer input_buffer;
		Py_buffer output_buffer;

		if (length_data) {
			for (int h = 0; h < batch_size; h++) {
				length_data[h] = boost::python::extract<int>(_length_data[h]);
			}
		}
		if (PyObject_GetBuffer(input.ptr(), &input_buffer, PyBUF_SIMPLE) != -1 && PyObject_GetBuffer(output.ptr(), &output_buffer, PyBUF_SIMPLE) != -1) {
			float ***input = new float**[NN->layer.front().size()];
			float ***output = new float**[NN->layer.back().size()];

			for (int i = 0, j = 0, index = 0; j < NN->layer[i].size(); j++) {
				input[j] = new float*[batch_size];

				for (int h = 0; h < batch_size; index += ((length_data) ? (length_data[h]) : ((NN->layer[i][j]->time_step) ? (NN->layer[i][j]->time_step) : (NN->time_step))) * NN->layer[i][j]->number_nodes, h++) {
					float *p = static_cast<float*>(input_buffer.buf);

					input[j][h] = &p[index];
				}
			}
			for (int i = NN->layer_height - 1, j = 0, index = 0; j < NN->layer[i].size(); j++) {
				output[j] = new float*[batch_size];

				for (int h = 0; h < batch_size; index += ((length_data) ? (length_data[h]) : ((NN->layer[i][j]->time_step) ? (NN->layer[i][j]->time_step) : (NN->time_step))) * NN->layer[i][j]->number_nodes, h++) {
					float *p = static_cast<float*>(output_buffer.buf);

					output[j][h] = &p[index];
				}
			}
			NN->Test(batch_size, input, output, length_data);

			for (int i = 0, j = 0; j < NN->layer[i].size(); j++) {
				delete[] input[j];
			}
			for (int i = NN->layer_height - 1, j = 0; j < NN->layer[i].size(); j++) {
				delete[] output[j];
			}
			delete[] input;
			delete[] output;

			PyBuffer_Release(&input_buffer);
			PyBuffer_Release(&output_buffer);
		}
		if (length_data) {
			delete[] length_data;
		}
	}

	double Train_CTC(int batch_size, int number_training, boost::python::object input, boost::python::list &reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list length_data;

		boost::python::object target_output;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train_CTC(int batch_size, int number_training, boost::python::list &length_data, boost::python::object input, boost::python::list &reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::object target_output;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, boost::python::object input, boost::python::object target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list length_data;
		boost::python::list reference;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, boost::python::list &length_data, boost::python::object input, boost::python::object target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list reference;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, boost::python::list &_length_data, boost::python::object input, boost::python::object target_output, boost::python::list &_reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		int *length_data = (len(_length_data)) ? (new int[number_training]) : (nullptr);

		double loss;

		vector<string> *reference = (len(_reference)) ? (new vector<string>[number_training]) : (nullptr);

		Py_buffer input_buffer;
		Py_buffer target_output_buffer;

		if (length_data) {
			for (int h = 0; h < number_training; h++) {
				length_data[h] = boost::python::extract<int>(_length_data[h]);
			}
		}
		if (reference) {
			for (int h = 0; h < number_training; h++) {
				for (int j = 0; j < len(_reference[h]); j++) {
					reference[h].push_back(boost::python::extract<string>(_reference[h][j]));
				}
			}
		}
		if (PyObject_GetBuffer(input.ptr(), &input_buffer, PyBUF_SIMPLE) != -1 && PyObject_GetBuffer(target_output.ptr(), &target_output_buffer, PyBUF_SIMPLE) != -1) {
			float ***input = new float**[NN->layer.front().size()];
			float ***target_output = new float**[NN->layer.back().size()];

			for (int i = 0, j = 0, index = 0; j < NN->layer[i].size(); j++) {
				input[j] = new float*[number_training];

				for (int h = 0; h < number_training; index += ((length_data) ? (length_data[h]) : ((NN->layer[i][j]->time_step) ? (NN->layer[i][j]->time_step) : (NN->time_step))) * NN->layer[i][j]->number_nodes, h++) {
					float *p = static_cast<float*>(input_buffer.buf);

					input[j][h] = &p[index];
				}
			}
			for (int i = NN->layer_height - 1, j = 0, index = 0; j < NN->layer[i].size(); j++) {
				target_output[j] = new float*[number_training];

				for (int h = 0; h < number_training; index += ((length_data) ? (length_data[h]) : ((NN->layer[i][j]->time_step) ? (NN->layer[i][j]->time_step) : (NN->time_step))) * NN->layer[i][j]->number_nodes, h++) {
					float *p = static_cast<float*>(target_output_buffer.buf);

					target_output[j][h] = &p[index];
				}
			}
			loss = NN->Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);

			for (int i = 0, j = 0; j < NN->layer[i].size(); j++) {
				delete[] input[j];
			}
			for (int i = NN->layer_height - 1, j = 0; j < NN->layer[i].size(); j++) {
				delete[] target_output[j];
			}
			delete[] input;
			delete[] target_output;

			PyBuffer_Release(&input_buffer);
			PyBuffer_Release(&target_output_buffer);
		}
		if (length_data) {
			delete[] length_data;
		}
		if (reference) {
			delete[] reference;
		}
		return loss;
	}
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Add_Layer, Neural_Networks_Wrapper::Add_Layer, 2, 6);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Initialize1, Neural_Networks_Wrapper::Initialize, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Initialize2, Neural_Networks_Wrapper::Initialize, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Test, Neural_Networks_Wrapper::Test, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train_CTC1, Neural_Networks_Wrapper::Train_CTC, 5, 7);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train_CTC2, Neural_Networks_Wrapper::Train_CTC, 6, 8);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train1, Neural_Networks_Wrapper::Train, 5, 7);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train2, Neural_Networks_Wrapper::Train, 6, 8);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train3, Neural_Networks_Wrapper::Train, 7, 9);

BOOST_PYTHON_MODULE(Neural_Networks) {
	using namespace boost::python;

	class_<Neural_Networks_Wrapper>("Neural_Networks", init<>())
	.def(init<int>())
	.def(init<string>())
	.def("Add_Layer", &Neural_Networks_Wrapper::Add_Layer, Add_Layer())
	.def("Connect", &Neural_Networks_Wrapper::Connect)
	.def("Initialize", static_cast<void(Neural_Networks_Wrapper::*)(double, double)>(&Neural_Networks_Wrapper::Initialize), Initialize1())
	.def("Initialize", static_cast<void(Neural_Networks_Wrapper::*)(int, double, double)>(&Neural_Networks_Wrapper::Initialize), Initialize2())
	.def("Save", &Neural_Networks_Wrapper::Save)
	.def("Set_Number_Threads", &Neural_Networks_Wrapper::Set_Number_Threads)
	.def("Set_Time_Mask", &Neural_Networks_Wrapper::Set_Time_Mask)
	.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(boost::python::object, boost::python::object, int)>(&Neural_Networks_Wrapper::Test), Test())
	.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(int, boost::python::object, boost::python::object)>(&Neural_Networks_Wrapper::Test))
	.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(int, boost::python::object, boost::python::object, boost::python::list&)>(&Neural_Networks_Wrapper::Test))
	.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::object, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train_CTC), Train_CTC1())
	.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, boost::python::object, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train_CTC), Train_CTC2())
	.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::object, boost::python::object, double, double, double)>(&Neural_Networks_Wrapper::Train), Train1())
	.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, boost::python::object, boost::python::object, double, double, double)>(&Neural_Networks_Wrapper::Train), Train2())
	.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, boost::python::object, boost::python::object, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train), Train3())
	;
}