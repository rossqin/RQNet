#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

ShuffleModule::ShuffleModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {

	groups = element->IntAttribute("groups", 3);
	if (groups <= 1) groups = 3;
	ParsePrevModules(element);
	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;
	ir_type = "ShuffleChannels";
	ir_params["data.group"] = to_string(groups);
	ir_params["data.axis"] = "1";
}

bool ShuffleModule::Resize(int w, int h) {
	input_width = output_width = w;
	output_height = input_height = h;
	return true;
}

bool ShuffleModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	char* src = (char*)(context.input->Data());
	char fullname[MAX_PATH];
	sprintf_s(fullname, MAX_PATH, "dbg\\%s.input", name.c_str());

	//context.input->DisplayInFile(fullname,1);  
	int group_channels = output_channels / groups;
	int fm_size = input_height * input_width * output.ElementBytes();

	for (int b = 0; b < output.Batch(); b++) {
		char* dest = output.BatchData(b);
		for (int src_c = 0; src_c < input_channels; src_c++) {
			int dest_c = (src_c % group_channels) * groups + (src_c / group_channels);
			if (cudaSuccess != cudaMemcpy(dest + dest_c * fm_size,
				src, fm_size, cudaMemcpyDeviceToDevice)) {
				cerr << " Error: Forward error in " << name << "! \n";
				return false;
			}
			src += fm_size;
		} 
	} 
	//sprintf_s(fullname, "dbg/%s.output", name.c_str());
	//output.DisplayInFile(fullname,1);
	return true;
}

bool ShuffleModule::Backward(CudaTensor& delta) {
	output.Release();
	if (!InferenceModule::Backward(delta)) return false;

	int group_channels = output_channels / groups;
	int fm_size = input_height * input_width * output.ElementBytes();

	CudaTensor copy = delta;

	for (int b = 0; b < delta.Batch(); b++) {
		char* src = copy.BatchData(b);
		char* dest = delta.BatchData(b);
		for (int src_c = 0; src_c < output_channels; src_c++) {
			int dest_c = (src_c % groups) * group_channels + (src_c / groups);
			if (cudaSuccess != cudaMemcpy(dest + dest_c * fm_size,
				src, fm_size, cudaMemcpyDeviceToDevice)) {
				cerr << " Error: Forward error in " << name << "! \n";
				return false;
			}
			src += fm_size;
		}

	}

	return DistributeDeltas(delta);
}
