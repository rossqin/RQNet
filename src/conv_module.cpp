#include "stdafx.h"
#include "cuda_tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"
#include <memory>

ConvolutionalModule::~ConvolutionalModule() {
	if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
	if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
}

ConvolutionalModule::ConvolutionalModule(const XMLElement * element,
	Layer * l, CNNNetwork* net, InferenceModule* prev) : InferenceModule(element, l,net, prev) ,
	w(net->DataType(),net->DataFormat()),
	dw(net->DataType(),net->DataFormat()),
	bias(net->DataType(), net->DataFormat()), 
	dbias(net->DataType(), net->DataFormat()),
	adam_m(net->DataType(), net->DataFormat()),
	adam_v(net->DataType(), net->DataFormat()),
	adam_bias_m(net->DataType(), net->DataFormat()),
	adam_bias_v(net->DataType(), net->DataFormat())
	{
	workspace_size = 0; 
	conv_desc = nullptr; 
	w_desc = nullptr;
	output_channels = element->IntAttribute("filters", 1);
	int width = 1, height = 1;
	int sz = element->IntAttribute("size", 0);

	if (sz > 0) {
		width = height = sz;
	}
	else {
		height = element->IntAttribute("filter-h", 1);
		width = element->IntAttribute("filter-w", 1);
	}
	
 
	bool hasBias = element->BoolAttribute("bias", false); 
	GetPrevModules(element);

	if (hasBias) {
		dbias.Init(1, output_channels, 1, 1);
		bias.Init(1, output_channels, 1, 1);
		bias.Randomize();		 
		network->weights_pool.Put(name + ".bias", &bias);
	}
	groups = element->IntAttribute("group", 1);
 
	 
		// TODO: ��һ�����groupsӦ�÷�������
	w.Init(output_channels, input_channels / groups, width, height);
	dw.Init(output_channels, input_channels / groups, width, height);
	 
	w.Randomize(); // in case we can't load weights
	
	
	network->weights_pool.Put(name + ".weights", &w);

	bool padding = element->BoolAttribute("padding", true);
	if (padding) {
		padding_w = element->IntAttribute("pad-w", (width - 1) >> 1);
		padding_h = element->IntAttribute("pad-h", (height - 1) >> 1);

	}
	else {
		padding_w = padding_h = 0;
	}
	int stride = element->IntAttribute("stride", 0);
	if (stride > 0) {
		stride_w = stride_h = stride;
	}
	else {
		stride_w = element->IntAttribute("stride-w", 1);
		stride_h = element->IntAttribute("stride-h", 1);
	}
	
	int dilation = element->IntAttribute("dilation", 0);
	if (dilation > 0) {
		dilation_w = dilation_h = dilation;
	}
	else {
		dilation_w = element->IntAttribute("dilation-w", 1);
		dilation_h = element->IntAttribute("dilation-h", 1);
	}
	
	cudnnCreateFilterDescriptor(&w_desc);
	if (w_desc) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetFilter4dDescriptor(w_desc, net->DataType(), net->DataFormat(),
			output_channels, input_channels / groups, height, width)) {
			cudnnDestroyFilterDescriptor(w_desc);
			w_desc = nullptr;
		}
	}
	cudnnCreateConvolutionDescriptor(&conv_desc);
	if (conv_desc) {
		if (groups > 1) {
			if (CUDNN_STATUS_SUCCESS != cudnnSetConvolutionGroupCount(conv_desc, groups)) {
				cudnnDestroyConvolutionDescriptor(conv_desc);
				conv_desc = nullptr;
			}
		}
		if (CUDNN_STATUS_SUCCESS != cudnnSetConvolution2dDescriptor(conv_desc,
			padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
			CUDNN_CROSS_CORRELATION, net->DataType())) {
			cudnnDestroyConvolutionDescriptor(conv_desc);
			conv_desc = nullptr;
		}
		
#if(CUDNN_MAJOR >= 7)
		// Tensor Core uses CUDNN_TENSOR_OP_MATH instead of CUDNN_DEFAULT_MATH
		// For *_ALGO_WINOGRAD_NONFUSED can be used CUDNN_DATA_FLOAT
		// otherwise Input, Filter and Output descriptors (xDesc, yDesc, wDesc, dxDesc, dyDesc and dwDesc as applicable) have dataType = CUDNN_DATA_HALF
		// Three techniques for training using Mixed-precision: https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
		// 1. Accumulation into FP32
		// 2. Loss Scaling - required only for: activation gradients. We do not use.
		// 3. FP32 Master Copy of Weights
		// More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops
		cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
		cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
#endif
#endif
	}

	fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	bwdd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	bwdf_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	forward_input = nullptr;
	if (GetAppConfig().UpdatePolicy()== Adam) {
		adam_m.Init(output_channels, input_channels, width, height);
		adam_v.Init(output_channels, input_channels, width, height);
		network->adam_weights_pool.Put(name + ".m", &adam_m);
		network->adam_weights_pool.Put(name + ".v", &adam_v);
		if (hasBias) {
			adam_bias_m.Init(1, output_channels, 1, 1);
			adam_bias_v.Init(1, output_channels, 1, 1);
			network->adam_weights_pool.Put(name + ".bias.m", &adam_bias_m);
			network->adam_weights_pool.Put(name + ".bias.v", &adam_bias_v);
		}
		

	}
	Resize(input_width, input_height); 
}

bool ConvolutionalModule::Forward(ForwardContext & context) { 
	if (!InferenceModule::Forward(context)) {
		return false;
	}
	
	
	float one = 1.0f, zero = 0.0f;

	if (context.input)
		forward_input = context.input;
	else
		forward_input = &input;
	CudaPtr<char> workspace(workspace_size);
	cudnnStatus_t status = cudnnConvolutionForward( GetCUDNNHandle(), &one,
		forward_input->Descriptor(), forward_input->Data(), w_desc, w, conv_desc, fwd_algo,
		workspace.ptr, workspace_size, &zero, output.Descriptor(), output);
	if (status != CUDNN_STATUS_SUCCESS) {
		cerr << "Error: Forwarding failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	bool ret = true;
	if (bias.Channel() == output_channels) {
		ret = output.Add(bias);
	}
	//input.Cache(cached_input);
	return ret;
}
bool ConvolutionalModule::Backward(CudaTensor& delta) {

	if (!InferenceModule::Backward(delta)) return false;
	cudnnHandle_t handle = GetCUDNNHandle();
	float one = 1.0f,zero = 0.0f;
	cudnnStatus_t status;
	if (bias.Channel() == output_channels) {
		status = cudnnConvolutionBackwardBias(handle, &one, delta.Descriptor(), delta, &one, dbias.Descriptor(), dbias);
		if (CUDNN_STATUS_SUCCESS != status) {
			cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl;
			return false;
		}
	} 
	CudaPtr<char> workspace(workspace_size);
	status = cudnnConvolutionBackwardFilter(handle, &one, forward_input->Descriptor(), forward_input->Data(),
		delta.Descriptor(), delta, conv_desc, bwdf_algo, workspace.ptr,
		workspace_size, &one, w_desc, dw);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl;
		return false;
	}
	CudaTensor temp = delta;
	if (!delta.Init(network->MiniBatch(), input_channels, input_height, input_width)) return false;
	status = cudnnConvolutionBackwardData(handle, &one, w_desc, w,
		temp.Descriptor(), temp, conv_desc, bwdd_algo, workspace,
		workspace_size, &zero, delta.Descriptor(), delta); 
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward data failed in`" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	// if(input.Elements() > 0) input.Release();
	return DistributeDeltas(delta);
}
extern bool sgd_update(void* params, void* updates, int elements, cudnnDataType_t data_type, float lr, bool decay);
extern bool adam_update(void* params, void* gt, void* mt, void* vt, int elements, int t, cudnnDataType_t data_type, float lr, bool decay);
bool ConvolutionalModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	if (cfg.ConvParamsFreezed()) {
		dw = 0.0f;
		dbias = 0.0f;
		return true;
	} 
	switch (cfg.UpdatePolicy()) {
	case SGD:
	{ 

		if (bias.Channel() == output_channels) {
			if (!sgd_update(bias, dbias, output_channels, bias.DataType(), lr, false)) return false;

		}
		return sgd_update(w, dw, w.Elements(), w.DataType(), lr, true);

	}
	case Adam: 
	{
		int t = network->cur_iteration;
		if (bias.Channel() == output_channels) {
			int t = network->cur_iteration;
			if (!adam_update(bias, dbias, adam_bias_m, adam_bias_v, bias.Elements(), t, bias.DataType(), lr, false)) return false;
		}
		return adam_update(w, dw, adam_m, adam_v, w.Elements(), t, w.DataType(), lr, true);
	}
	default:
		return true;
	}
	return true;
}

bool ConvolutionalModule::Resize(int w, int h) {
	if (!conv_desc || ! w_desc) return false;
	int b = network->MiniBatch();

	cudnnTensorDescriptor_t x_desc = input.Descriptor();

	if (!x_desc && forward_input) {
		x_desc = forward_input->Descriptor();
	}
	cudnnTensorDescriptor_t y_desc = output.Descriptor();
	bool created_x = false, created_y = false;
	if (!x_desc) {
		cudnnCreateTensorDescriptor(&x_desc);
		cudnnSetTensor4dDescriptor(x_desc, network->DataFormat(), network->DataType(), b, input_channels, h, w);		
		created_x = true;
	}
	if (!y_desc) {
		cudnnCreateTensorDescriptor(&y_desc);
		created_y = true;
	}


	int c;
	cudnnStatus_t err = cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &b, &c, &output_height, &output_width);

	if (CUDNN_STATUS_SUCCESS != err)
		return false;
	//TODO: assert(c == output_channels)
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc, output.DataFormat(), output.DataType(), b, output_channels, output_height, output_width))
		return false;

	cudnnHandle_t handle = GetCUDNNHandle();
#if(CUDNN_MAJOR <= 7)
	cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc,
		conv_desc, y_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo);
	cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc,
		conv_desc, x_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwdd_algo);
	cudnnGetConvolutionBackwardFilterAlgorithm(handle, x_desc,
		y_desc, conv_desc, w_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwdf_algo);
#endif
	//*****/
	size_t w1 = 0, w2 = 0, w3 = 0;
	cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &w1);
	cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, bwdd_algo, &w2);
	cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, bwdf_algo, &w3);
	if (w1 < w2) w1 = w2;
	if (w1 < w3) w1 = w3;
	workspace_size = w1; 
	if (created_x) {
		cudnnDestroyTensorDescriptor(x_desc);
	}
	if(created_y){
		cudnnDestroyTensorDescriptor(y_desc);
	}
	input_width = w;
	input_height = h;
	return true;
}

/*
Description: group denotes the number of groups to which output and input should be split. 
For example, 
group equal 1 means that all the filters are applied to full input (usual convolution), 
group equals 2 means that both input and output channels are separated into 2 groups 
and i-th output group is connected to i-th input group channels. 
group equals number of output feature maps denotes depth-wise separable convolution
*/
bool ConvolutionalModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int &l_index) const {
	

	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset,l_index)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"Convolution\">" << endl;
	xml << "      <data auto_pad=\"same_upper\" dilations=\"" << dilation_h << "," << dilation_w
		<< "\" group=\"1\" kernel=\"" << w.Height() << "," << w.Width() << "\" output=\"" << output_channels
		<< "\" pads_begin=\"" << padding_h << "," << padding_w << "\" pads_end=\"" << padding_h << "," << padding_w
		<< "\" strides=\"" << stride_h << "," << stride_w << "\"/>" << endl;
	WritePorts(xml);
	xml << "      <blobs>" << endl;
 
	
	CpuPtr<char> w_cpu(w.Bytes(), w.Data());
	CpuPtr<char> bias_cpu(bias.Bytes(), bias.Data()); 
	if (prevs.size() == 0) {
		// the first module, we need to divide 255
		if (w.DataType() == CUDNN_DATA_FLOAT) {
			float* pf = reinterpret_cast<float*>(w_cpu.ptr);
			for (int i = 0; i < w.Elements(); i++) {
				pf[i] = pf[i] / 255.0f;
			}
		}
		else {
			__half* ph = reinterpret_cast<__half*>(w_cpu.ptr);
			float t;
			for (int i = 0; i < w.Elements(); i++) { 
				t = __half2float(ph[i]) / 255.0f;				
				ph[i] = __float2half(t) ;
			}

		}
	}
	
// 	float *display = reinterpret_cast<float*>(w_cpu.ptr);
// 	__half temp;
// 	char line[100];
// 	unsigned int ui;
// 	unsigned short us;
// 	for (int i = 0; i < w.Elements(); i++) {
// 		temp = __float2half(display[i]);
// 		ui = *(reinterpret_cast<unsigned int *>(display + i));
// 		us = *(reinterpret_cast<unsigned short *>(&temp));
// 		sprintf(line, "%03d : 0x%08x(%.6f) - 0x%04x\n", i, ui, display[i], us);
// 		
// 		cout << line;
// 		
// 	}
	
	/*
	Weights layout is GOIYX (GOIZYX for 3D convolution),
	which means that X is changing the fastest, then Y, then Input, Output, then Group.
	*/
	bin.write(w_cpu.ptr, w.Bytes());
	xml << "        <weights offset=\"" << bin_offset << "\"  size=\"" << w.Bytes() << "\" />" << endl;
	bin_offset += w.Bytes();

	bin.write(bias_cpu.ptr, bias.Bytes());
	xml << "        <biases offset=\"" << bin_offset << "\"  size=\"" << bias.Bytes() << "\" />" << endl;
	bin_offset += bias.Bytes();

	xml << "      </blobs>" << endl;
	xml << "    </layer>" << endl;
	 
	return true;
}
uint32_t ConvolutionalModule::GetFlops() const {
	return 0;
}