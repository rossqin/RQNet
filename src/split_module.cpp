#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

SplitModule::SplitModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {

	ParsePrevModules(element);
	groups = element->IntAttribute("groups", 1);

	int axis = element->IntAttribute("axis", 1);

	if (axis != 1) {
		throw exception("Only support axis=\"1\" in split module !\n");
	}
	output_channels = input_channels / groups;
	
	output_height = input_height;
	output_width = input_width;
	ir_params["data.axis"] = "1";
	ir_params["data.num_split"] = to_string(groups);
	ir_type = "Split";

	CudaTensor temp(network->DataType(), network->DataFormat());
	if (!temp.Init({ network->MiniBatch(),output_channels, output_height, output_width })) {
		throw exception("SplitModule module creation failed due to tensor failed");
	}
	for (int g = 0; g < groups; g++) {
		outputs.push_back(temp);
		deltas.push_back(temp);
	}
	forward_input = nullptr;
}

bool SplitModule::Resize(int w, int h) {
	input_width = w;
	input_height = h;

	output_width = w;
	output_height = h;
	for (int g = 0; g < groups; g++) {
		if (!outputs[g].Init({ network->MiniBatch(), output_channels, output_height, output_width })) return false;
		if (!deltas[g].Init({ network->MiniBatch(), output_channels, output_height, output_width })) return false;
	}
	return true;
}

bool SplitModule::Forward(ForwardContext& context) {

	if (prevs.size() == 0) return false;
	if (!InferenceModule::Forward(context)) return false;
	forward_input = context.input;
	size_t batch_size = output_channels * output_height * output_width * forward_input->ElementBytes();
	size_t offset = 0;
	for (int g = 0; g < groups; g++) {
		CudaTensor& o = outputs[g];
		for (int b = 0; b < network->MiniBatch(); b++) {
			if (cudaSuccess != cudaMemcpy(o.BatchData(b), forward_input->BatchData(b) + offset,
				batch_size, cudaMemcpyDeviceToDevice)) {
				cerr << "Error: " << name << ".forward failed due to memory failed !\n";
				return false;
			}
		}
		offset += batch_size;
	}
 
	return true;
}

bool SplitModule::Backward(CudaTensor& delta) { 
	if (!delta.Init(forward_input->Dims())) {
		cerr << "Error: " << name << ".backward failed due to delta initialization failed! \n";
		return false;
	}
	size_t batch_size = output_channels * output_height * output_width * forward_input->ElementBytes();
	size_t offset = 0;
	for (int g = 0; g < groups; g++) {
		CudaTensor& d = deltas[g];
		for (int b = 0; b < network->MiniBatch(); b++) {
			if (cudaSuccess != cudaMemcpy(delta.BatchData(b) + offset, d.BatchData(b), batch_size, cudaMemcpyDeviceToDevice)) {
				cerr << "Error: " << name << ".forward failed due to memory failed !\n";
				return false;
			}
		}
		offset += batch_size; 
		deltas[g] = 0.0f;
	}

	input.Release();
	forward_input = nullptr;
	return DistributeDeltas(delta);
}

bool SplitModule::ShortcutDelta(const CudaTensor& d, int group_id) {
	if(group_id < 0 || group_id >= deltas.size() ) return false;
	return deltas[group_id].Add(d); 
}
