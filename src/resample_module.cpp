#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

UpSampleModule::UpSampleModule(const XMLElement* element, Layer* l, CNNNetwork* network, InferenceModule* prev) :
	InferenceModule(element, l, network, prev) {

	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2);

	ParsePrevModules(element);
	output_width = input_width * stride_w;
	output_height = input_height * stride_h;
	output_channels = input_channels;
	ir_type = "Resample";
	char buffer[32];
	sprintf_s(buffer, 32, "%u", stride_h);
	ir_params["data.factor"] = buffer;	
	ir_params["data.type"] = "caffe.ResampleParameter.NEAREST";// myriad only supports caffe.ResampleParameter.NEAREST
}

UpSampleModule::~UpSampleModule() {
}
bool UpSampleModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width * stride_w;
	output_height = input_height * stride_h;
	return true;
}
bool UpSampleModule::Forward(ForwardContext& context) {
	
	if (!InferenceModule::Forward(context)) return false; 
	if (!context.input->UpSample(output, stride_w, stride_h)) {
		cerr << "Error: " << name << ".forward failed!\n";
	}
	return true;
}

bool UpSampleModule::Backward(CudaTensor& delta) {
	if (!InferenceModule::Backward(delta)) return false;
	CudaTensor temp(network->DataType(), network->DataFormat());
	temp = delta;
	if (!delta.Init({ network->MiniBatch(),input_channels, input_height, input_width })) {
		return false;
	}
	if (!temp.DownSample(delta, stride_w, stride_h)) return false;
	output.Release();
	
	return DistributeDeltas(delta);
}

uint32_t UpSampleModule::GetFlops() const {
	return 0;
}