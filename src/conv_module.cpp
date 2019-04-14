#include "stdafx.h"
#include "tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"
#include <memory>

ConvolutionalModule::~ConvolutionalModule() {
	if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
	if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
	if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
	if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
	if (db_desc) cudnnDestroyTensorDescriptor(db_desc);
}

ConvolutionalModule::ConvolutionalModule(const XMLElement * element,
	Layer * l, InferenceModule* prev) : InferenceModule(element, l, prev) {
	db_desc = NULL;
	conv_desc = NULL;
	w_desc = NULL;
	followed_bn_module = NULL;
	output_channels = element->IntAttribute("filters", 1);
	int width = element->IntAttribute("filter-w", 1);
	int height = element->IntAttribute("filter-h", 1);
	bool hasBias = element->BoolAttribute("bias", false);
	TensorOrder order = GetNetwork().GetDataOrder();
	cudnnTensorFormat_t cudnn_order = (order == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	GetPrevModules(element);

	if (hasBias) {
		dbias.Init(1, output_channels, 1, 1, order);
		bias = dbias;
		GetParamPool().Put(name + ".bias", &bias);
		cudnnCreateTensorDescriptor(&db_desc);
		if (db_desc) {
			cudnnSetTensor4dDescriptor(db_desc, cudnn_order, CUDNN_DATA_FLOAT, 1, output_channels, 1, 1);
		}
	}

	cudnnCreateTensorDescriptor(&x_desc);
	cudnnCreateTensorDescriptor(&y_desc);
	cudnnCreateFilterDescriptor(&w_desc);

	cudnnCreateConvolutionDescriptor(&conv_desc);


	w.Init(output_channels, input_channels, width, height, order);
	dw.Init(output_channels, input_channels, width, height, order);
	if (w_desc != NULL) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, cudnn_order,
			output_channels, input_channels, height, width)) {
			cudnnDestroyFilterDescriptor(w_desc);
			w_desc = NULL;
		}
	}
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

	if (conv_desc != NULL) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetConvolution2dDescriptor(conv_desc,
			padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
			CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)) {
			cudnnDestroyConvolutionDescriptor(conv_desc);
			conv_desc = NULL;
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
	InitDescriptors();
}

bool ConvolutionalModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	float one = 1.0f, zero = 0.0f;
	cudnnStatus_t status = cudnnConvolutionForward( GetCUDNNHandle(), &one,
		x_desc, input.GetMem(), w_desc, w.GetMem(), conv_desc, fwd_algo,
		GetNetwork().workspace, GetNetwork().workspace_size, &zero, y_desc, output.GetMem());
	if (status != CUDNN_STATUS_SUCCESS) {
		cerr << "Error: Forwarding failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	if (bias.GetChannels() != 0) {
		return output.Add(bias);
	}
	return true;
}
bool ConvolutionalModule::Backward(FloatTensor4D & delta) {

	if (!InferenceModule::Backward(delta)) return false;
	cudnnHandle_t handle = GetCUDNNHandle();
	float one = 1.0f,zero = 0.0f;
	if (bias.GetChannels() != 0) {
		cudnnConvolutionBackwardBias(handle, &one, y_desc, delta.GetMem(), &one, db_desc, dbias.GetMem());
	}
	cudnnStatus_t status = cudnnConvolutionBackwardFilter(handle, &one, x_desc, input.GetMem(),
		y_desc, delta.GetMem(), conv_desc, bwdf_algo, GetNetwork().workspace,
		GetNetwork().workspace_size, &one, w_desc, dw.GetMem());
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl;
		return false;
	}
	FloatTensor4D temp = delta;
	if (!delta.InitFrom(input)) return false;
	status = cudnnConvolutionBackwardData(handle, &one, w_desc, w.GetMem(),
		y_desc, temp.GetMem(), conv_desc, bwdd_algo, GetNetwork().workspace,
		GetNetwork().workspace_size, &zero, x_desc, delta.GetMem());
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward data failed in`" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	return DistributeDeltas(delta);
}

bool ConvolutionalModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	if (cfg.ConvParamsFreezed()) {
		dw = 0.0f;
		dbias = 0.0f;
		return true;
	}
	float m = cfg.Momentum();
	float decay = (0.0f - cfg.Decay()) *  cfg.GetBatch();
	if (cfg.UpdateStrategy() == "SGD") {
		if (bias.GetChannels() != 0) {
			if (!dbias.AddScale(bias, decay)) return false;
			if (!bias.AddScale(dbias, lr)) return false;
			if (!dbias.Mul(m)) return false;
		}

		if (!dw.AddScale(w, decay)) return false;
		if (!w.AddScale(dw, lr)) return false;
		if (!dw.Mul(m)) return false;

	}
	dw = 0.0f;
	dbias = 0.0f;
	return true;
}

bool ConvolutionalModule::InitDescriptors() {
	if (NULL == x_desc || NULL == y_desc || NULL == conv_desc || NULL == w_desc) return false;
	int b = GetAppConfig().GetMiniBatch();

	cudnnTensorFormat_t f = (input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc, f, CUDNN_DATA_FLOAT, b, input_channels, input_height, input_width))
		return false;
	int c;
	if (CUDNN_STATUS_SUCCESS != cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &b, &c, &output_height, &output_width))
		return false;
	//TODO: assert(c == output_channels)
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc, f, CUDNN_DATA_FLOAT, b, output_channels, output_height, output_width))
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
	GetNetwork().UpdateWorkspace(w1);
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
bool ConvolutionalModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const {
	
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, fp16)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"Convolution\">" << endl;
	xml << "      <data auto_pad=\"same_upper\" dilations=\"" << dilation_h << "," << dilation_w
		<< "\" group=\"1\" kernel=\"" << w.GetHeight() << "," << w.GetWidth() << "\" output=\"" << output_channels
		<< "\" pads_begin=\"" << padding_h << "," << padding_w << "\" pads_end=\"" << padding_h << "," << padding_w
		<< "\" strides=\"" << stride_h << "," << stride_w << "\"/>" << endl;
	WritePorts(xml);
	xml << "      <blobs>" << endl;
	int length_w = 0;
	int length_b = 0;
	FloatTensor4D tensor_w = w;
	FloatTensor4D tensor_b;
	if (!tensor_b.Init(1, output_channels, 1, 1, TO_NCHW)) return false;

	 
	unique_ptr<char> bias_cpu(new char[output_channels * sizeof(float)]);
	unique_ptr<char> w_cpu(new char[tensor_w.MemBytes()]);
	if (followed_bn_module) {
		if(!followed_bn_module->CalcWeightsForIR(tensor_w, tensor_b))
			return false;
	}
	else {	 
		if (bias.MemElements() == output_channels) {
			tensor_b = bias;
		}
	}

	if (fp16) {
		length_w = tensor_w.MemElements() * sizeof(__half);
		CudaPtr<__half> temp(tensor_w.MemElements());
		if (!f32_to_f16(temp, tensor_w, tensor_w.MemElements())) return false;
		if(!temp.ToCPU(w_cpu.get())) return false;

		length_b = output_channels * sizeof(__half);
		if (!f32_to_f16(temp, tensor_b, output_channels)) return false;
		if (!temp.ToCPU(bias_cpu.get(), length_b)) return false;
	}
	else {
		length_w = tensor_w.MemBytes();
		length_b = tensor_b.MemBytes();
		cudaMemcpy(w_cpu.get(), tensor_w, length_w, cudaMemcpyDeviceToHost);
		cudaMemcpy(bias_cpu.get(), tensor_b, length_b, cudaMemcpyDeviceToHost);
	}
	/*
	Weights layout is GOIYX (GOIZYX for 3D convolution),
	which means that X is changing the fastest, then Y, then Input, Output, then Group.
	*/
	bin.write(w_cpu.get(), length_w);
	xml << "        <weights offset=\"" << bin_offset << "\"  size=\"" << length_w << "\" />" << endl;
	bin_offset += length_w;

	bin.write(bias_cpu.get(), length_b);
	xml << "        <biases offset=\"" << bin_offset << "\"  size=\"" << length_b << "\" />" << endl;
	bin_offset += length_b;

	xml << "      </blobs>" << endl;
	xml << "    </layer>" << endl;

	
	 
	return true;
}
uint32_t ConvolutionalModule::GetFlops() const {
	return 0;
}