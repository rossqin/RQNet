#include "stdafx.h"
#include "inference_module.h"
bool SSDModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height;

	return false;
}
SSDModule::SSDModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {
	ParsePrevModules(element);
	focal_loss = element->BoolAttribute("focal-loss", true);
	int k = element->IntAttribute("k", true); 
	// by default 2.0 / sk = (w + h );
	s_k = 2.0 / (output_height + output_width);

	ignore_thresh = element->FloatAttribute("ignore-thresh", 0.5f);
	truth_thresh = element->FloatAttribute("truth-thresh", 0.7f);
	
}

SSDModule::~SSDModule()
{
}

bool SSDModule::Forward(ForwardContext & context)
{
	return false;
}

bool SSDModule::Backward(CudaTensor & delta)
{
	return false;
}

 