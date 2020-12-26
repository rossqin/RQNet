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
	adam_bias_v(net->DataType(), net->DataFormat()) {
	workspace_size = 0; 
	conv_desc = nullptr; 
	w_desc = nullptr;
	output_channels = element->IntAttribute("filters", input_channels);
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
	ParsePrevModules(element);

	if (hasBias) {		
		bias.Init({ 1, output_channels, 1, 1 });
		bias.Randomize();
		dbias.Init(bias.Dims());
		network->weights_pool.Put(name + ".bias", &bias);
	}
	if (_strcmpi("dwconv", element->Attribute("type")) == 0) {
		groups = input_channels;
	}
	else 
		groups = element->IntAttribute("groups", 1);
	 
	w.Init({ output_channels, input_channels / groups, width, height });
	dw.Init(w.Dims());
	 
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
		adam_m.Init(w.Dims());
		adam_v.Init(w.Dims());
		network->adam_weights_pool.Put(name + ".m", &adam_m);
		network->adam_weights_pool.Put(name + ".v", &adam_v);
		if (hasBias) {
			adam_bias_m.Init(bias.Dims());
			adam_bias_v.Init(bias.Dims());
			network->adam_weights_pool.Put(name + ".bias.m", &adam_bias_m);
			network->adam_weights_pool.Put(name + ".bias.v", &adam_bias_v);
		}
		

	}
	Resize(input_width, input_height); 
	ir_type = "Convolution";
}

bool ConvolutionalModule::Forward(ForwardContext & context) { 
	if (!InferenceModule::Forward(context)) {
		return false;
	} 
	float one = 1.0f, zero = 0.0f; 
	forward_input = context.input; 
	CudaPtr<char> workspace((int)workspace_size);
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
	CudaPtr<char> workspace((int)workspace_size);
	status = cudnnConvolutionBackwardFilter(handle, &one, forward_input->Descriptor(), forward_input->Data(),
		delta.Descriptor(), delta, conv_desc, bwdf_algo, workspace.ptr,
		workspace_size, &one, w_desc, dw);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl;
		return false;
	}
	CudaTensor temp = delta;
	if (!delta.Init({ network->MiniBatch(), input_channels, input_height, input_width })) return false;
	status = cudnnConvolutionBackwardData(handle, &one, w_desc, w,
		temp.Descriptor(), temp, conv_desc, bwdd_algo, workspace,
		workspace_size, &zero, delta.Descriptor(), delta); 
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward data failed in`" << name << "`. Error code :" << (int)status << endl;
		return false;
	}

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

bool ConvolutionalModule::Resize(int w_, int h) {
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
		cudnnSetTensor4dDescriptor(x_desc, network->DataFormat(), network->DataType(), b, input_channels, h, w_);		
		created_x = true;
	}
	if (!y_desc) {
		cudnnCreateTensorDescriptor(&y_desc);
		created_y = true;
	}
	if (groups > 1 && groups > input_channels) {
		groups = input_channels;
		if (CUDNN_STATUS_SUCCESS != cudnnSetConvolutionGroupCount(conv_desc, groups)) {
			cudnnDestroyConvolutionDescriptor(conv_desc);
			conv_desc = nullptr;
		}
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
	input_width = w_;
	input_height = h;
	return true;
}
const float MAX_FP16 = 65000.0f;
bool ConvolutionalModule::RenderOpenVINOIR(vector<OpenVINOIRv7Layer>& layers, vector<OpenVINOIRv7Edge>& edges, 
	ofstream& bin, size_t& bin_offset, bool fp16) {

	CpuPtr<float> w_cpu(w.Elements() , w.Data());
	CpuPtr<float> bias_cpu(bias.Elements(), bias.Data());

	ir_params.clear(); 

	char buffer[32]; 
	// this leads to wrong result when strides=2
	//ir_params["data.auto_pad"] = "same_upper"; 

	sprintf_s(buffer, 32, "%u,%u", dilation_h, dilation_w);
	ir_params["data.dilations"] = buffer;

	sprintf_s(buffer, 32, "%u", groups);
	ir_params["data.group"] = buffer;

	sprintf_s(buffer, 32, "%u,%u", w.Height(), w.Width());
	ir_params["data.kernel"] = buffer;

	sprintf_s(buffer, 32, "%u", output_channels);
	ir_params["data.output"] = buffer;

	sprintf_s(buffer, 32, "%u,%u", padding_h, padding_w);
	ir_params["data.pads_begin"] = buffer;
	ir_params["data.pads_end"] = buffer;

	sprintf_s(buffer, 32, "%u,%u", stride_h, stride_w);
	ir_params["data.strides"] = buffer;  
	if (fp16) { 
		CpuPtr<__half> data(w_cpu.Length() + bias_cpu.Length());

		int i = 0;
		while (i < w_cpu.Length()) {
			//
			float f = w_cpu.ptr[i];
			// range of fp16: -65504~ -6.10 * 10e-5, 6.10 * 10e-5 ~ -65504
			if (f > MAX_FP16) f = MAX_FP16;
			else if(f < -MAX_FP16) f = -MAX_FP16; 
			data.ptr[i] = __float2half(f);
			i++;
		}
		for (int j = 0; j < bias_cpu.Length(); j++, i++) {
			float f = bias_cpu.ptr[j];
			if (f > MAX_FP16) f = MAX_FP16;
			else if (f < -MAX_FP16) f = -MAX_FP16;
			data.ptr[i] = __float2half(f);
		}
		bin.write(reinterpret_cast<char*>(data.ptr), data.Length() * 2);

		sprintf_s(buffer, 32, "%lu", (unsigned long)bin_offset);
		ir_params["weights.offset"] = buffer;
		sprintf_s(buffer, 32, "%lu", (unsigned long)(w_cpu.Length() * 2));
		ir_params["weights.size"] = buffer;
		bin_offset += (w_cpu.Length() * 2); 
		if (bias.Bytes() > 0) {
			sprintf_s(buffer, 32, "%lu", (unsigned long)bin_offset);
			ir_params["zbiases.offset"] = buffer;
			sprintf_s(buffer, 32, "%lu", (unsigned long)(bias_cpu.Length() * 2));
			ir_params["zbiases.size"] = buffer; 
			bin_offset += (bias_cpu.Length() * 2);
		}
	}
	else { // FP32
		bin.write(reinterpret_cast<char*>(w_cpu.ptr), w.Bytes());

		sprintf_s(buffer, 32, "%lu", (unsigned long)bin_offset);
		ir_params["weights.offset"] = buffer;
		sprintf_s(buffer, 32, "%lu", (unsigned long)w.Bytes());
		ir_params["weights.size"] = buffer;
		bin_offset += w.Bytes();
		if (bias.Bytes() > 0) {
			bin.write(reinterpret_cast<char*>(bias_cpu.ptr), bias.Bytes());
			sprintf_s(buffer, 32, "%lu", (unsigned long)bin_offset);
			ir_params["zbiases.offset"] = buffer;
			sprintf_s(buffer, 32, "%lu", (unsigned long)bias.Bytes());
			ir_params["zbiases.size"] = buffer;
			bin_offset += bias.Bytes();
		}
	}
	return InferenceModule::RenderOpenVINOIR(layers,edges,bin,bin_offset, fp16);
}
uint32_t ConvolutionalModule::GetFlops() const {
	return 0;
}

bool prune_params(CudaTensor& params, const vector<bool>& v, bool input) {
	CpuPtr<float> saved(params.Elements(), params.Data());
	int prune_count = 0;
	for (bool b : v) {
		if (b) prune_count++;
	}
	int c_size = params.Height() * params.Width();
	int b_size = params.Channel() * c_size;
	bool ret = true;
	if (input) {
		int saved_channels = params.Channel();
		if (!params.Init({ params.Batch(),saved_channels - prune_count , params.Height(), params.Width() })) {
			cerr << "Params reinitialization failed!\n";
			return false;
		}
		float* buffer = New float[params.Elements()];
		for (int b = 0; b < params.Batch(); b++) {
			float* dst = buffer + b * params.Channel() * c_size;
			float* src = saved.ptr + b * b_size;
			for (int sc = 0; sc < saved_channels; sc++, src += c_size) {
				if (!v[sc]) {
					memcpy(dst, src, c_size * sizeof(float));
					dst += c_size;
				}
			}
		}
		ret = params.Push(buffer, 0, params.Elements());
		delete[]buffer;
	}
	else {
		int saved_batch = params.Batch();
		if (!params.Init({ params.Batch() - prune_count, params.Channel(), params.Height(), params.Width() })) {
			cerr << "Params reinitialization failed!\n";
			return false;
		}
		float* buffer = New float[params.Elements()];
		float* dst = buffer;
		float* src = saved.ptr;
		for (int b = 0; b < saved_batch; b++, src += b_size) {
			if (!v[b]) {
				memcpy(dst, src, b_size * sizeof(float));
				dst += b_size;
			}
		}
		ret = params.Push(buffer, 0, params.Elements());
		delete[]buffer;
	}
	return ret;
}
/***/


bool ConvolutionalModule::PruneChannel(vector<PruneInfo>& prune_infos, float threshold) {
	if (!InferenceModule::PruneChannel(prune_infos, threshold)) return false;
	int kw = w.Width();
	int kh = w.Height();
	// Check wheter input_channels has changed.
	int expected_wc = input_channels / groups;
	bool ic_changed = (expected_wc != w.Channel());
	if (input_channels == groups) {
		if (input_channels != output_channels) {
			ic_changed = true; // for dwconv
			output_channels = input_channels;
		}
	}

	int c_size = w.Height() * w.Width();
	if (prune_infos.size() > 0) {
		if (ic_changed) {
			// find the match prune info
			int g = -1;
			InferenceModule* m = GetPrev(0, g, false);
			vector<bool>* v = nullptr;
			while (m) {
				for (auto& p : prune_infos) {
					if (p.module == m) {
						v = &(p.channels_removed);
						break;
					}
				}
				if (v) break;
				m = m->GetPrev(0, g, false);
			}
			if (m) {
				if (!v) {
					cerr << "Error: Depthwise convolution module `" << name << "` cannot be pruned!\n";
					return false;
				}
				//w.DisplayInFile((name + ".w.init.txt").c_str());
				if (groups == input_channels) { // dwconv, 
					if (!prune_params(w, *v, false)) return false;
				}
				else if (groups == 1) { //normal conv
					if (!prune_params(w, *v, true)) return false;
				}
				else {
					cerr << "Error: Group convolution module `" << name << "` has not been supported yet!\n";
					return false;
				} 
				
				if (groups == input_channels) {
					cout << " INFO: The input channels and filters of `" << name << "` set to " << input_channels << "!\n";
					return true;
				}
				else
					cout << " INFO: The input channels of `" << name << "` set to " << input_channels << "!\n";
		 

			}
		}
	}

	PruneInfo info;
	info.module = this;
	info.removed_channel_count = 0;
	bool pruned = false;
	info.channels_removed.assign(output_channels, pruned);

	CpuPtr<float> weights_in_cpu(w.Elements(), w.Data());

	for (int i = 0; i < output_channels; i++) {
		float sq_sum = 0.0f;
		for (int j = 0;  j < expected_wc; j++) {
			int n = (i * expected_wc + j) * c_size;
			for (int k = 0; k < c_size; k++, n++) {
				float f = weights_in_cpu.ptr[n];
				sq_sum += f * f;
			}
		}
		if (sq_sum < threshold) {
			info.channels_removed[i] = true;
			info.removed_channel_count++;
		}
	}
	if (info.removed_channel_count > 0) {
		//w.DisplayInFile((name + ".weights.before.txt").c_str());
		if (!prune_params(w, info.channels_removed, false)) return false;
		//w.DisplayInFile((name + ".weights.post.txt").c_str());
		if (bias.Elements() > 0) {
			//bias.DisplayInFile((name + ".bias.before.txt").c_str());
			if (!prune_params(bias, info.channels_removed, true)) return false;
			//bias.DisplayInFile((name + ".bias.post.txt").c_str());
		}
		output_channels -= info.removed_channel_count;
		cout << " INFO: The filters value of `" << name << "` set to " << output_channels << "!\n";
		prune_infos.push_back(info);
	}


	return true;
}

