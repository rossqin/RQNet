#include "stdafx.h"
#include "network.h"
#include "inference_module.h"

bool EltwiseModule::Resize(int w, int h) {
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

EltwiseModule::EltwiseModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {

	operation = SUM;
	const char* op_str = element->Attribute("operation");
	if (op_str) {
		if (0 == _strcmpi(op_str, "sub")) {
			operation = SUB;
		}
		else if (0 == _strcmpi(op_str, "mul")) {
			operation = MUL;
		}
		else if (0 == _strcmpi(op_str, "div")) {
			operation = DIV;
		}
		else if (0 == _strcmpi(op_str, "max")) {
			operation = MAX;
		}
		else if (0 == _strcmpi(op_str, "min")) {
			operation = MIN;
		}
		else if (0 == _strcmpi(op_str, "squared_diff")) {
			operation = SQUARED_DIFF;
		}
		else if (0 == _strcmpi(op_str, "floor_mod")) {
			operation = FLOOR_MOD;
		}
		else if (0 == _strcmpi(op_str, "pow")) {
			operation = POW;
		}
		else if (0 == _strcmpi(op_str, "logical_and")) {
			operation = LOGICAL_AND;
		}
		else if (0 == _strcmpi(op_str, "logical_or")) {
			operation = LOGICAL_OR;
		}
		else if (0 == _strcmpi(op_str, "logical_xor")) {
			operation = LOGICAL_XOR;
		}
		else if (0 == _strcmpi(op_str, "less")) {
			operation = LESS;
		}
		else if (0 == _strcmpi(op_str, "less_equal")) {
			operation = LESS_EQUAL;
		}
		else if (0 == _strcmpi(op_str, "greater")) {
			operation = GREATER;
		}
		else if (0 == _strcmpi(op_str, "greater_equal")) {
			operation = GREATER_EQUAL;
		}
		else if (0 == _strcmpi(op_str, "equal")) {
			operation = EQUAL;
		}
		else if (0 == _strcmpi(op_str, "not_equal")) {
			operation = NOT_EQUAL;
		}
		ir_params["operation"] = op_str;
	}
	else {
		ir_params["operation"] = "SUM";
	}
	concat_prevs = false;
	ParsePrevModules(element);
	output_height = input_height;
	output_width = input_width;
	output_channels = input_channels;
	if (prevs.size() == 1) {
		prevs.push_back({ prev,-1 });
	}
	ir_type = "Eltwise";

}

bool EltwiseModule::Forward(ForwardContext& context) { 
	if (prevs.size() < 2) {
		cerr << "Error: " << name << " requires more than 1 previous modules!\n";
		return false;
	}
	int w = 0, h = 0;
	for (auto& p : prevs) {
		CudaTensor& o = p.module->GetOutput(p.group_id);
		if (o.Height() > h) h = o.Height();
		if (o.Width() > w) w = o.Width();
	}
	if (w != output_width) 	output_width = w;
	if (h != output_height) output_height = h;

	int expected_output_elemens = network->MiniBatch() * output_channels * output_height * output_width;
	if (output.Elements() != expected_output_elemens &&
		!output.Init({ network->MiniBatch(), output_channels, output_height, output_width })) {
		cerr << "Error: " << name << ".forward failed due to output initialization failed!\n";
		return false;
	}
	else
		output = 0.0f;
	for (auto& p : prevs) {
		CudaTensor& o = p.module->GetOutput(p.group_id);

		//TODO: surport more than ADD
		if (!output.Add(o)) return false; 
	}
	 
	return true;
}

bool EltwiseModule::Backward(CudaTensor& delta) {
	if (!InferenceModule::Backward(delta)) {
		cerr << "Error: " << name << ".backward failed due to shortcut delta error!\n";
		return false;
	}
	for (int n = 0; n < prevs.size(); n++) {
		InferenceModule* m = prevs[n].module;		 
		if (!m->ShortcutDelta(delta, prevs[n].group_id)) {
			cerr << "Error: " << name << ".backward.DistributeDeltas failed !\n";
			return false;
		} 
	}
	return delta.Release();
}
 

