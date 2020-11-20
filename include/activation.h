#pragma once
enum ActivationMode {
	LOGISTIC, LINEAR, LEAKY, RELU, HARDTAN, LHTAN,  TANH, ELU, LOGGY,RELU6
};
bool gradient_array_ongpu(const void* y, void* delta, int elements, cudnnDataType_t data_type, ActivationMode mode);
bool activate_array_ongpu(const void* x, void* y, int elements, cudnnDataType_t data_type, ActivationMode mode);
