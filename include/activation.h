#pragma once
enum ACTIVATION_TYPE {
	LOGISTIC, LINEAR, LEAKY, RELU, HARDTAN, LHTAN, RELIE, RAMP, TANH, PLSE, ELU, LOGGY, STAIR
};
bool gradient_array_ongpu(float *x, float * delta, int n, ACTIVATION_TYPE a);
bool activate_array_ongpu(float *x, int n, ACTIVATION_TYPE a);
