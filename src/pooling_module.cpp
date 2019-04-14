#include "stdafx.h"
#include "tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"

bool PoolingModule::InitDescriptors() {

	output_width = (input_width + pad_w - window_w) / stride_w + 1;
	output_height = (input_height + pad_h - window_h) / stride_h + 1;
	//if (!output.Init(b, c, output_width, output_height, input.GetOrder())) return false; 
	return true;
}

PoolingModule::PoolingModule(const XMLElement * element, Layer * l, InferenceModule* prev) :
	InferenceModule(element, l, prev) {
 
	indexes = NULL;
	const char* s = element->Attribute("type");
	string t(s ? s : "max-pool");
	window_w = element->IntAttribute("size-w", 2);
	window_h = element->IntAttribute("size-h", 2);

	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2);

	pad_w = element->IntAttribute("pad-w", window_w - 1);
	pad_h = element->IntAttribute("pad-h", window_h - 1);

	GetPrevModules(element);

	mode = POOLING_MAX;
	if (t == "avg-pool") mode = POOLING_AVG;
	InitDescriptors();

	output_channels = input_channels;
}

PoolingModule::~PoolingModule() {
	if (indexes) cudaFree(indexes);
}

extern bool forward_maxpool(FloatTensor4D& output, const FloatTensor4D& input, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h);
extern bool backward_maxpool(FloatTensor4D& prev_delta, const FloatTensor4D& delta, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h);

bool PoolingModule::Forward(ForwardContext & context) {
	int saved_h = input_height;
	int saved_w = input_width;

	if (!InferenceModule::Forward(context)) {
		cerr << name << " failed! \n";
		return false;
	}	

	if (context.training && (saved_h != input_height || saved_w != input_width)) {
		if (indexes) {
			cudaFree(indexes);
			indexes = NULL;
		}
		cudaMalloc(&indexes, output.MemElements() * sizeof(int));
	}

	if (!forward_maxpool(output, input, indexes, window_w, window_h, pad_w, pad_h, stride_w, stride_h)) {
		cerr << name << " failed! \n";
		return false;
	}
	return true;
}

bool PoolingModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	if (!backward_maxpool(input, delta, indexes, window_w, window_h, pad_w, pad_h, stride_w, stride_h))
		return false;
	delta = input;
	return DistributeDeltas(delta);
}
bool PoolingModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, fp16)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"Pooling\">" << endl;
	xml << "      <data auto_pad=\"valid\" exclude-pad=\"true\" kernel=\"" << window_h << "," << window_w;
	if (stride_h == 1 && stride_w == 1) { // f
		xml << "\" pads_begin=\"0,0\" pads_end=\"1,1\" pool-method=\"max\" strides=\""
			<< stride_h << "," << stride_w << "\" />" << endl;
	}
	else {
		xml << "\" pads_begin=\"0,0\" pads_end=\"0,0\" pool-method=\"max\" strides=\""
			<< stride_h << "," << stride_w << "\" />" << endl;
	}
	WritePorts(xml); 
	xml << "    </layer>" << endl;
	return true;
}
uint32_t PoolingModule::GetFlops() const {
	return 0;
}