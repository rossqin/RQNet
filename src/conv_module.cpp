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
	dbias(net->DataType(), net->DataFormat()){
	 
	conv_desc = nullptr; 
	w_desc = nullptr;
	output_channels = element->IntAttribute("filters", 1);
	int width = element->IntAttribute("filter-w", 1);
	int height = element->IntAttribute("filter-h", 1);
	bool hasBias = element->BoolAttribute("bias", false); 
	GetPrevModules(element);

	if (hasBias) {
		dbias.Init(1, output_channels, 1, 1);
		bias = dbias;
		GetParamPool().Put(name + ".bias", &bias);
	}

	w.Init(output_channels, input_channels, width, height);
	dw.Init(output_channels, input_channels, width, height);
	
	GetParamPool().Put(name + ".weights", &w);

	bool padding = element->BoolAttribute("padding", true);
	if (padding) {
		padding_w = element->IntAttribute("pad-w", (width - 1) >> 1);
		padding_h = element->IntAttribute("pad-h", (height - 1) >> 1);

	}
	else {
		padding_w = 0;
		padding_h = 0;
	}
	stride_w = element->IntAttribute("stride-w", 1);
	stride_h = element->IntAttribute("stride-h", 1);

	dilation_w = element->IntAttribute("dilation-w", 1);
	dilation_h = element->IntAttribute("dilation-h", 1);
	
	cudnnCreateFilterDescriptor(&w_desc);
	if (w_desc) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetFilter4dDescriptor(w_desc, net->DataType(), net->DataFormat(),
			output_channels, input_channels, height, width)) {
			cudnnDestroyFilterDescriptor(w_desc);
			w_desc = nullptr;
		}
	}
	cudnnCreateConvolutionDescriptor(&conv_desc);
	if (conv_desc) {
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
	Resize(input_width, input_height);
}

bool ConvolutionalModule::Forward(ForwardContext & context) {
	if (name == "layer03.convolution") {
		cout << "checkpoint\n";
	}
	if (!InferenceModule::Forward(context)) return false;
	float one = 1.0f, zero = 0.0f;

	cudnnStatus_t status = cudnnConvolutionForward( GetCUDNNHandle(), &one,
		input.Descriptor(), input , w_desc, w, conv_desc, fwd_algo,
		network->workspace, network->workspace_size, &zero, output.Descriptor(), output);
	if (status != CUDNN_STATUS_SUCCESS) {
		cerr << "Error: Forwarding failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	bool ret = true;
	if (bias.Channel() == output_channels) {
		ret = output.Add(bias);
	}
	//output.Save(DEBUGGING_DIR "conv02.output.bin", 1);
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
	status = cudnnConvolutionBackwardFilter(handle, &one, input.Descriptor(), input,
		delta.Descriptor(), delta, conv_desc, bwdf_algo, network->workspace,
		network->workspace_size, &one, w_desc, dw);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl;
		return false;
	}
	CudaTensor temp = delta;
	if (!delta.Init(input.Batch(), input_channels, input_height, input_width)) return false;
	status = cudnnConvolutionBackwardData(handle, &one, w_desc, w,
		temp.Descriptor(), temp, conv_desc, bwdd_algo, network->workspace,
		network->workspace_size, &zero, delta.Descriptor(), delta); 
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward data failed in`" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	return DistributeDeltas(delta);
}
extern bool sgd_update(void* params, void* updates, int elements, cudnnDataType_t data_type, float lr, float decay, float momentum);

bool ConvolutionalModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	if (cfg.ConvParamsFreezed()) {
		dw = 0.0f;
		dbias = 0.0f;
		return true;
	}
	float m = cfg.Momentum();
	float decay = (0.0f - cfg.Decay()) *  cfg.GetBatch();
	lr /= cfg.GetBatch();
	if (cfg.UpdateStrategy() == "SGD") {
		if (bias.Channel() == output_channels) {
			if(!sgd_update(bias, dbias, output_channels, bias.DataType(), lr , decay, m)) return false;

		}
		if (!sgd_update(w, dw, w.Elements(), w.DataType(), lr, decay, m)) return false; 

	} 
	return true;
}

bool ConvolutionalModule::Resize(int w, int h) {
	if (!conv_desc || ! w_desc) return false;
	int b = network->MiniBatch();

	cudnnTensorDescriptor_t x_desc = input.Descriptor();
	cudnnTensorDescriptor_t y_desc = output.Descriptor();
	bool created = false;
	if (!x_desc) {
		cudnnCreateTensorDescriptor(&x_desc);
		cudnnSetTensor4dDescriptor(x_desc, input.DataFormat(), input.DataType(), b, input_channels, h, w);
		cudnnCreateTensorDescriptor(&y_desc);
		created = true;
	}

	int c;
	if (CUDNN_STATUS_SUCCESS != cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &b, &c, &output_height, &output_width))
		return false;
	//TODO: assert(c == output_channels)
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc, output.DataFormat(), output.DataType(), b, output_channels, output_height, output_width))
		return false;

	cudnnHandle_t handle = GetCUDNNHandle();
	cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc,
		conv_desc, y_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo);
	cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, y_desc,
		conv_desc, x_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwdd_algo);
	cudnnGetConvolutionBackwardFilterAlgorithm(handle, x_desc,
		y_desc, conv_desc, w_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwdf_algo);

	size_t w1 = 0, w2 = 0, w3 = 0;
	cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &w1);
	cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, y_desc, conv_desc, x_desc, bwdd_algo, &w2);
	cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, y_desc, conv_desc, w_desc, bwdf_algo, &w3);
	if (w1 < w2) w1 = w2;
	if (w1 < w3) w1 = w3;
	network->UpdateWorkspace(w1);
	if (created) {
		cudnnDestroyTensorDescriptor(x_desc);
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
bool ConvolutionalModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset) const {
	
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"Convolution\">" << endl;
	xml << "      <data auto_pad=\"same_upper\" dilations=\"" << dilation_h << "," << dilation_w
		<< "\" group=\"1\" kernel=\"" << w.Height() << "," << w.Width() << "\" output=\"" << output_channels
		<< "\" pads_begin=\"" << padding_h << "," << padding_w << "\" pads_end=\"" << padding_h << "," << padding_w
		<< "\" strides=\"" << stride_h << "," << stride_w << "\"/>" << endl;
	WritePorts(xml);
	xml << "      <blobs>" << endl;
 
	
	CpuPtr<char> w_cpu(w.Bytes(), w.Data());
	CpuPtr<char> bias_cpu(bias.Bytes(), bias.Data()); 
 
	
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