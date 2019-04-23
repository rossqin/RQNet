#include "stdafx.h"
#include "cuda_tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"

bool PoolingModule::Resize(int w, int h) {
	int b = network->MiniBatch();
	cudnnTensorDescriptor_t x_desc = input.Descriptor();
	bool created = false;
	input_height = h;
	input_width = w;

	output_width = (input_width + pad_wl + pad_wr - window_w) / stride_w + 1;
	output_height = (input_height + pad_ht + pad_hb - window_h) / stride_h + 1;

	if (indexes) {
		cudaFree(indexes);
		indexes = nullptr;
	}
	return cudaSuccess == cudaMalloc(&indexes, network->MiniBatch() * output_channels * output_height * output_width * sizeof(int));
 
}

PoolingModule::PoolingModule(const XMLElement * element, Layer * l,CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {
 
	indexes = nullptr;
	const char* s = element->Attribute("type");
	string t(s ? s : "max-pool");
	window_w = element->IntAttribute("size-w", 2);
	window_h = element->IntAttribute("size-h", 2);

	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2);
 
	pad_wl = 0;
	pad_ht = 0;
	pad_wr = 0;
	pad_hb = 0;
	const char* pads_begin = element->Attribute("pads_begin");
	if (pads_begin) { 
		vector<string> strs;
		split_string(strs, pads_begin); 
		if (strs.size() == 2) {
			pad_ht = atoi(strs[0].c_str());
			pad_wl = atoi(strs[1].c_str());
		}
	}
	const char* pads_end = element->Attribute("pads_end");
	if (pads_end) {
		vector<string> strs;
		split_string(strs, pads_end);
		if (strs.size() == 2) {
			pad_hb = atoi(strs[0].c_str());
			pad_wr = atoi(strs[1].c_str());
		}
	}

	GetPrevModules(element);

	mode = CUDNN_POOLING_MAX;
	if (t == "avg-pool") mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	output_channels = input_channels;	 
	Resize(input_width, input_width);
	
}

PoolingModule::~PoolingModule() { 
	if (indexes) cudaFree(indexes);
}
extern bool forward_maxpool(CudaTensor& output, const CudaTensor& input, int* indexes,
	int window, int stride , int pad);
bool PoolingModule::Forward(ForwardContext & context) { 
	if (!InferenceModule::Forward(context)) {
		cerr << " Error: " << name << " foward failed! \n";
		return false;
	}
	if (mode != CUDNN_POOLING_MAX) return false;
	
	int window = (window_h << 16) + (window_w & 0xffff);
	int stride = (stride_h << 16) + (stride_w & 0xffff); 
	int pad = (pad_wl << 24) + ((pad_wr & 0xff) << 16) + ((pad_ht & 0xff) << 8) + (pad_hb & 0xff);
 
	return forward_maxpool(output,input,indexes, window, stride, pad);
}
extern bool backward_maxpool(CudaTensor& dx, const CudaTensor& dy, int* indexes, int window, int stride, int pad);
bool PoolingModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	CudaTensor temp = delta;
	if (!delta.Init(network->MiniBatch(), input_channels, input_height, input_width)) return false;
	int window = (window_h << 16) + (window_w & 0xffff);
	int stride = (stride_h << 16) + (stride_w & 0xffff);
	int pad = (pad_wl << 24) + ((pad_wr & 0xff) << 16) + ((pad_ht & 0xff) << 8) + (pad_hb & 0xff);
	if (!backward_maxpool(delta, temp, indexes, window, stride, pad)) return false;	
	return DistributeDeltas(delta);
}
bool PoolingModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset,l_index)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"Pooling\">" << endl;
	xml << "      <data auto_pad=\"valid\" exclude-pad=\"true\" kernel=\"" << window_h << "," << window_w;
 
	xml << "\" pads_begin=\""<< pad_ht <<"," << pad_wl <<"\" pads_end=\"" << pad_hb << "," << pad_wr << "\" pool-method=\"max\" strides=\""
			<< stride_h << "," << stride_w << "\" />" << endl;
	 
	WritePorts(xml); 
	xml << "    </layer>" << endl;
	return true;
}
uint32_t PoolingModule::GetFlops() const {
	return 0;
}