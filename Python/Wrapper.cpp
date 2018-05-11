#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <iostream>
#include <omp.h>

#include "../Neural_Networks.h"

class Float {
private:
	int depth;
	int height;
	int width;
public:
	float ***memory;

	Float() {
		memory = nullptr;
	}
	Float(boost::python::list &list) {
		int dimension = (static_cast<string>(boost::python::extract<string>(list[0].attr("__class__").attr("__name__"))) != "list") ? (1) : ((static_cast<string>(boost::python::extract<string>(list[0][0].attr("__class__").attr("__name__"))) != "list") ? (2) : (3));

		switch (dimension) {
		case 1:
			depth = 1;
			height = 1;
			width = len(list);
			break;
		case 2:
			depth = 1;
			height = len(list);
			width = len(list[0]);
			break;
		case 3:
			depth = len(list);
			height = len(list[0]);
			width = len(list[0][0]);
		}

		memory = new float**[depth];

		for (int z = 0; z < depth; z++) {
			memory[z] = new float*[height];

			for (int y = 0; y < height; y++) {
				memory[z][y] = new float[width];
			}
		}

		switch (dimension) {
		case 1:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						memory[z][y][x] = boost::python::extract<float>(list[x]);
					}
				}
			}
			break;
		case 2:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						memory[z][y][x] = boost::python::extract<float>(list[y][x]);
					}
				}
			}
			break;
		case 3:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						memory[z][y][x] = boost::python::extract<float>(list[z][y][x]);
					}
				}
			}
		}
	}
	~Float() {
		if (memory) {
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					delete[] memory[z][y];
				}
				delete[] memory[z];
			}
			delete[] memory;
		}
	}

	void ToList(boost::python::list &list) {
		int dimension = (static_cast<string>(boost::python::extract<string>(list[0].attr("__class__").attr("__name__"))) != "list") ? (1) : ((static_cast<string>(boost::python::extract<string>(list[0][0].attr("__class__").attr("__name__"))) != "list") ? (2) : (3));

		switch (dimension) {
		case 1:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						list[x] = memory[z][y][x];
					}
				}
			}
			break;
		case 2:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						list[y][x] = memory[z][y][x];
					}
				}
			}
			break;
		case 3:
			for (int z = 0; z < depth; z++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						list[z][y][x] = memory[z][y][x];
					}
				}
			}
		}
	}
};

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
	void Test(Float &input, boost::python::list &output, int _length_data = 0) {
		boost::python::list length_data;

		if (_length_data) {
			length_data.append(_length_data);
		}
		Test(1, input, output, length_data);
	}
	void Test(int batch_size, Float &input, boost::python::list &output) {
		boost::python::list length_data;

		Test(batch_size, input, output, length_data);
	}
	void Test(int batch_size, Float &input, boost::python::list &_output, boost::python::list &_length_data) {
		int *length_data = (len(_length_data)) ? (new int[batch_size]) : (nullptr);

		Float output = Float(_output);

		if (length_data) {
			for (int h = 0; h < batch_size; h++) {
				length_data[h] = boost::python::extract<int>(_length_data[h]);
			}
		}
		NN->Test(batch_size, input.memory, output.memory, length_data);
		output.ToList(_output);

		if (length_data) {
			delete[] length_data;
		}
	}

	double Train_CTC(int batch_size, int number_training, Float &input, boost::python::list &reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list length_data;

		Float target_output;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train_CTC(int batch_size, int number_training, boost::python::list &length_data, Float &input, boost::python::list &reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		Float target_output;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, Float &input, Float &target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list length_data;
		boost::python::list reference;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, boost::python::list &length_data, Float &input, Float &target_output, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		boost::python::list reference;

		return Train(batch_size, number_training, length_data, input, target_output, reference, learning_rate, epsilon, noise_standard_deviation);
	}
	double Train(int batch_size, int number_training, boost::python::list &_length_data, Float &input, Float &target_output, boost::python::list &_reference, double learning_rate, double epsilon = 0, double noise_standard_deviation = 0) {
		int *length_data = (len(_length_data)) ? (new int[number_training]) : (nullptr);

		double loss;

		vector<string> *reference = (len(_reference)) ? (new vector<string>[number_training]) : (nullptr);

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
		loss = NN->Train(batch_size, number_training, length_data, input.memory, target_output.memory, reference, learning_rate, epsilon, noise_standard_deviation);

		if (length_data) {
			delete[] length_data;
		}
		if (reference) {
			delete[] reference;
		}
		return loss;
	}
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Add_Layer, Add_Layer, 2, 6);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Initialize1, Initialize, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Initialize2, Initialize, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Test, Neural_Networks_Wrapper::Test, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train_CTC1, Neural_Networks_Wrapper::Train_CTC, 5, 7);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train_CTC2, Neural_Networks_Wrapper::Train_CTC, 6, 8);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train1, Neural_Networks_Wrapper::Train, 5, 7);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train2, Neural_Networks_Wrapper::Train, 6, 8);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Train3, Neural_Networks_Wrapper::Train, 7, 9);

BOOST_PYTHON_MODULE(Neural_Networks) {
	using namespace boost::python;

	class_<Float>("Float", init<>())
		.def(init<boost::python::list&>())
		;

	class_<Neural_Networks_Wrapper>("Neural_Networks", init<>())
		.def(init<int>())
		.def(init<string>())
		.def("Add_Layer", &Neural_Networks_Wrapper::Add_Layer, Add_Layer())
		.def("Connect", &Neural_Networks_Wrapper::Connect)
		.def("Initialize", static_cast<void(Neural_Networks_Wrapper::*)(double, double)>(&Neural_Networks_Wrapper::Initialize), Initialize1())
		.def("Initialize", static_cast<void(Neural_Networks_Wrapper::*)(int, double, double)>(&Neural_Networks_Wrapper::Initialize), Initialize2())
		.def("Save", &Neural_Networks_Wrapper::Save)
		.def("Set_Number_Threads", &Neural_Networks_Wrapper::Set_Number_Threads)
		.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(Float&, boost::python::list&, int)>(&Neural_Networks_Wrapper::Test), Test())
		.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(int, Float&, boost::python::list&)>(&Neural_Networks_Wrapper::Test))
		.def("Test", static_cast<void(Neural_Networks_Wrapper::*)(int, Float&, boost::python::list&, boost::python::list&)>(&Neural_Networks_Wrapper::Test))
		.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, Float&, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train_CTC), Train_CTC1())
		.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, Float&, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train_CTC), Train_CTC2())
		.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, Float&, Float&, double, double, double)>(&Neural_Networks_Wrapper::Train), Train1())
		.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, Float&, Float&, double, double, double)>(&Neural_Networks_Wrapper::Train), Train2())
		.def("Train", static_cast<double(Neural_Networks_Wrapper::*)(int, int, boost::python::list&, Float&, Float&, boost::python::list&, double, double, double)>(&Neural_Networks_Wrapper::Train), Train3())
		;
}