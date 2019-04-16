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
	if (!x_desc) {
		cudnnCreateTensorDescriptor(&x_desc);
		cudnnSetTensor4dDescriptor(x_desc, input.DataFormat(), input.DataType(), b, input_channels, h, w);
		created = true;
	}
	int c;
	if(CUDNN_STATUS_SUCCESS != cudnnGetPooling2dForwardOutputDim(desc, x_desc, &b, &c, &output_height, &output_width))
		return false;
	if (!created && stride_w == 1 && stride_h == 1) {
		if (!stride_one_output.Init(b, c, input_height, input_width)) return false;
	}
	input_height = h;
	input_width = w;
	if (created)
		cudnnDestroyTensorDescriptor(x_desc);
	return true;
}

PoolingModule::PoolingModule(const XMLElement * element, Layer * l,CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) , stride_one_output(net->DataType(), net->DataFormat()){
 
	 
	desc = nullptr;
	cudnnCreatePoolingDescriptor(&desc);
	const char* s = element->Attribute("type");
	string t(s ? s : "max-pool");
	window_w = element->IntAttribute("size-w", 2);
	window_h = element->IntAttribute("size-h", 2);

	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2);

	int def_pad_w = (stride_w % 2) ? 0 : 1;
	int def_pad_h = (stride_h % 2) ? 0 : 1;

	pad_w = element->IntAttribute("pad-w", def_pad_w);
	pad_h = element->IntAttribute("pad-h", def_pad_h);

	GetPrevModules(element);

	mode = CUDNN_POOLING_MAX;
	if (t == "avg-pool") mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	Resize(input_width, input_width);
	cudnnSetPooling2dDescriptor(desc, mode, CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, pad_h, pad_w, stride_h, stride_w);
	output_channels = input_channels;
}

PoolingModule::~PoolingModule() {
	if (desc) cudnnDestroyPoolingDescriptor(desc);
}
const float one = 1.0f, zero = 0.0f;
extern bool one_stride_pooling_patch(CudaTensor& out, const CudaTensor& in, bool forwarding);
bool PoolingModule::Forward(ForwardContext & context) {
	int saved_h = input_height;
	int saved_w = input_width;

	if (!InferenceModule::Forward(context)) {
		cerr << " Error: " << name << " foward failed! \n";
		return false;
	}
	
	cudnnStatus_t status = cudnnPoolingForward(GetCUDNNHandle(), desc, &one, input.Descriptor(), input, &zero, output.Descriptor(), output);
	if (status != CUDNN_STATUS_SUCCESS) {
		return false;
	}
	if (stride_w == 1 && stride_h == 1) {
		if (!one_stride_pooling_patch(stride_one_output, output,true)) return false;
		context.input = &stride_one_output;
	}
	
	/*
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
	}*/

	return true;
}

bool PoolingModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	CudaTensor temp(delta.DataType(), delta.DataFormat());
	if (!temp.Init(output.Batch(), output_channels, output_height, output_width)) return false;
	if (stride_w == 1 && stride_h == 1) {		
		if (!one_stride_pooling_patch(temp, delta, true)) return false; 
	}
	else {
		if (cudaSuccess != cudaMemcpy(temp, delta, temp.Bytes(), cudaMemcpyDeviceToDevice)) return false;
		if (!delta.Init(input.Batch(), input_channels, input_height, input_width)) return false;

	}
	if (CUDNN_STATUS_SUCCESS != cudnnPoolingBackward(GetCUDNNHandle(), desc, &one, output.Descriptor(), output,
		temp.Descriptor(), temp, input.Descriptor(), input, &zero, delta.Descriptor(), delta))
		return false;
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