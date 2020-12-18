#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

ConcatModule::ConcatModule(const XMLElement* element, Layer* l,
	CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {
	ParsePrevModules(element);
	output_height = input_height;
	output_width = input_width;
	output_channels = input_channels;
	ir_type = "Concat";
}
bool ConcatModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height;
	return true;
}

bool ConcatModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	output = *(context.input);
	return true;
}

bool ConcatModule::Backward(CudaTensor& delta) {
	if (!InferenceModule::Backward(delta)) return false;
	return DistributeDeltas(delta);
}