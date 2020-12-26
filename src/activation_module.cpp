#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

bool ActivationModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height;

	// for channels pruning 
	if (output_channels != input_channels) {
		cout << "Input channels of `" << name << "` has changed, change ouput_channels to fit.\n";
		output_channels = input_channels;
	}

	return true;
}

ActivationModule::ActivationModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {


	factor = 0.1f;
	const char* a = element->Attribute("method");
	if (!a) a = net->DefaultActivation().c_str();
	//string str(a ? a : );
	ir_type = "Activation";
	if (0 == _strcmpi(a, "leaky")) {
		mode = LEAKY;
		ir_type = "ReLU";
		ir_params["data.negative_slope"] = "0.1";
	}
	else if (0 == _strcmpi(a, "linear")) mode = LINEAR;
	else if (0 == _strcmpi(a, "logistic")) {
		mode = LOGISTIC;
		ir_params["data.type"] = "sigmoid";
	}
	else if (0 == _strcmpi(a, "relu")) mode = RELU;
	else if (0 == _strcmpi(a, "lhtan")) mode = LHTAN;
	else if (0 == _strcmpi(a, "hardtan")) mode = HARDTAN;
	else if (0 == _strcmpi(a, "tanh")) {
		mode = TANH;
		ir_params["data.type"] = "tanh";
	}
	else if (0 == _strcmpi(a, "loggy")) mode = LOGGY;
	else if (0 == _strcmpi(a, "elu")) {
		mode = ELU;
		ir_params["data.type"] = "elu";
	}
	else if (0 == _strcmpi(a, "relu6")) {
		mode = RELU6;
		ir_params["data.type"] = "relu6";
	}
	else if (0 == _strcmpi(a, "mish")) {
		mode = MISH;
		ir_type = "Mish";
	}
	else {
		mode = LEAKY;
		ir_type = "ReLU";
	}
	ParsePrevModules(element);
	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;

}


bool ActivationModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	void* data = context.input ? context.input->Data() : input.Data();
	bool r = activate_array_ongpu(data, output, output.Elements(), output.DataType(), mode);

	return r;
}
bool ActivationModule::Backward(CudaTensor& delta) {
	if (!InferenceModule::Backward(delta)) return false;
	if (!gradient_array_ongpu(output, delta, output.Elements(), output.DataType(), mode)) return false;
	return DistributeDeltas(delta);
}
uint32_t ActivationModule::GetFlops() const {
	return 0;
}
