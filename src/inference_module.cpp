#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "tensor.h"
#include "yolo.h"
#include "inference_module.h"
#include "config.h"
const float one = 1.0f, zero = 0.0f;
void InferenceModule::GetPrevModules(const XMLElement* element) {
	string base = layer->GetName() + ".";
	const char* id = element->Attribute("id");
	if (NULL == id) id = "convolution";
	name = base + id;
	ModulePool& module_pool = GetModulePool();	
	const char *p = element->Attribute("before");	 
	string prev_ids_s(p ? p : "none");	
		 
	if (prev_ids_s.length() > 0 && prev_ids_s != "none" ) {
		vector<string> prev_ids;
		ModulePool::iterator  it;
		split_string(prev_ids, prev_ids_s);
		input_channels = 0;
		for (auto id : prev_ids) {
			if (id.find('.') == string::npos)
				it = module_pool.find(base + id);
			else
				it = module_pool.find(id);
			if (it != module_pool.end()) {
				input_channels += it->second->GetOutputChannels();
				prevs.push_back(it->second);
			}
		} 
	}
	else
		input_channels = GetNetwork().input.GetChannels();
}
 
InferenceModule* InferenceModule::FromXmlElement(const XMLElement* element,  Layer* layer, TensorOrder order, InferenceModule* prev) {
	const char *t = element->Attribute("type");
	string mtype(t ? t : "");
	InferenceModule* module = NULL;
	if (mtype == "conv" || mtype == "convolutional") {
		module = New ConvolutionalModule(element, layer, order, prev);
	}
	else if (mtype == "batch-norm" ) {
		module = New BatchNormModule(element, layer, order, prev);
	}
	else if(mtype == "activation") {
		module = New ActivationModule(element, layer, order, prev);
	}
	else if (mtype == "max-pool" || mtype == "avg-pool") {
		module = New PoolingModule(element, layer, order, prev);
	}
	else if (mtype == "upsample") {
		module = New UpSampleModule(element, layer, order, prev);
	}
	else if (mtype == "yolo-detection") {
		module = New YoloModule(element, layer, order, prev);
	}
	//TODO: Add your New Moule here.

	if (module)
		GetModulePool().insert(pair<string, InferenceModule*>(module->name, module));
	return module;
}
bool InferenceModule::PrepareShortcutDelta() {
	if (shortcut_delta.SameDememsion(output))
		return true;
	return shortcut_delta.InitFrom(output);
}
bool InferenceModule::UpdateShortcutDelta(const FloatTensor4D& delta) {
	if (shortcut_delta.MemElements() == 0) {
		shortcut_delta = delta;
		return true;
	}
	if (shortcut_delta.SameDememsion(delta)) {
		return shortcut_delta.Add(delta);
	}
	return false;
}
bool InferenceModule::Forward(ForwardContext & context) {
	int n = prevs.size();
	if (n == 1) {
		input = prevs[0]->output;
	}
	else if (n > 1) {
		input = 0.0f;
		int batch = GetAppConfig().GetMiniBatch();
		int w = 0, h = 0;
		TensorOrder to = GetNetwork().GetDataOrder();
		for (int i = 0; i < n; i++) {
			if (prevs[i]->output.GetHeight() > h)
				h = prevs[i]->output.GetHeight();
			if (prevs[i]->output.GetWidth() > w)
				w = prevs[i]->output.GetWidth();
		} 

		if (!input.Init(batch, input_channels,w, h, to)) return false; 
		for (int b = 0; b < batch; b++) { 
			for (int i = 0, c = 0; i < n; i++) {
				FloatTensor4D& o = prevs[i]->output;
				size_t elements = o.Elements3D();
				float* src = o.GetMem() + b * elements;	
				if(!input.Concat(b, c, src, o.GetChannels(), o.GetWidth(),o.GetHeight()))
					return false;
				c += o.GetChannels(); 
			}
		}		
	}
	else
		input = context.input;
	if (input_width != input.GetWidth() || input_height != input.GetHeight()) {
		if (!InitDescriptors(context.training)) return false;
		input_width = input.GetWidth();
		input_height = input.GetHeight();
	}
	
	return true;
}
 
bool InferenceModule::Backward(FloatTensor4D & delta) {
	if (delta.MemElements() == 0 && shortcut_delta.MemElements() != 0) {
		delta = shortcut_delta;
		return shortcut_delta.Release();
	}
	if (delta.SameDememsion(shortcut_delta)) {
		if(!delta.Add(shortcut_delta)) return false;
		return shortcut_delta.Release();
	}
	return true;
}
bool InferenceModule::DistributeDeltas(FloatTensor4D & delta) {
	int n = prevs.size();
	if (n == 0) return true;
	if (n < 2) {
		InferenceModule* module = prevs[0];
		if (module != logical_prev) {			 
			return module->shortcut_delta.Add(delta);
		}
		return true;
	}
	for (int i = 0, c = 0; i < n; i++) {
		InferenceModule* module = prevs[i];
		int c_prev = module->output.GetChannels();
		if (!module->PrepareShortcutDelta()) return false;
		for (int b = 0; b < delta.GetBatch(); b++) {
			float* src = delta.GetMem() + (b * delta.GetChannels() + c) * delta.Elements2D();
			if (!module->shortcut_delta.Concat(b, 0, src, c_prev,  delta.GetWidth(), delta.GetHeight()))
				return false;
		}
		c += c_prev;
	} 
	return delta.Release(); 
}
ConvolutionalModule::~ConvolutionalModule() {
	if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
	if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
	if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
	if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
	if (db_desc) cudnnDestroyTensorDescriptor(db_desc);
}

ConvolutionalModule::ConvolutionalModule(const XMLElement * element, Layer * l, TensorOrder order, InferenceModule* prev)  {
	input = NULL;
	db_desc = NULL;
	layer = l;
	input_width = 0;
	input_height = 0;
	logical_prev = prev;
	output_channels = element->IntAttribute("filters", 1);
	int width = element->IntAttribute("filter-w", 1);
	int height = element->IntAttribute("filter-h", 1); 
	bool hasBias = element->BoolAttribute("bias", false);
	GetPrevModules(element);

	if (hasBias) {
		dbias.Init(1, output_channels, 1, 1, GetNetwork().GetDataOrder());
		bias = dbias;
		GetParamPool().Put(name + ".bias", &bias);
		cudnnCreateTensorDescriptor(&db_desc);
		if (db_desc) {
			cudnnSetTensor4dDescriptor(db_desc,
				(bias.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
				CUDNN_DATA_FLOAT, dbias.GetBatch(), output_channels, 1, 1);
		}
	}

	x_desc = NULL;
	cudnnCreateTensorDescriptor(&x_desc);
	y_desc = NULL;
	cudnnCreateTensorDescriptor(&y_desc);
	w_desc = NULL;
	cudnnCreateFilterDescriptor(&w_desc);
	conv_desc = NULL;
	cudnnCreateConvolutionDescriptor(&conv_desc);
	
	
	w.Init(output_channels, input_channels, width, height, order);
	dw.Init(output_channels, input_channels, width, height, order);
	if (w_desc != NULL) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetFilter4dDescriptor(w_desc,
			CUDNN_DATA_FLOAT, (order == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
			output_channels, input_channels, height, width)) {
			cudnnDestroyFilterDescriptor(w_desc);
			w_desc = NULL;
		}
	}  
	GetParamPool().Put(name + ".weights", &w);

	bool padding = element->BoolAttribute("padding", true);
	if (padding) {
		padding_w = element->IntAttribute("pad-w", (width - 1 ) >> 1 );
		padding_h = element->IntAttribute("pad-h", (height - 1) >> 1 );

	}
	else {
		padding_w = 0;
		padding_h = 0;
	}
	stride_w = element->IntAttribute("stride-w", 1);
	stride_h = element->IntAttribute("stride-h", 1);

	int dilation_w = element->IntAttribute("dilation-w", 1);
	int dilation_h = element->IntAttribute("dilation-h", 1);

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
	input_height = 0;
	input_width = 0;
}

bool ConvolutionalModule::Forward(ForwardContext & context) { 
	if (!InferenceModule::Forward(context)) return false; 
	
	cudnnStatus_t status = cudnnConvolutionForward(
		GetCUDNNHandle(),
		&one,
		x_desc,
		input.GetMem(),
		w_desc,
		w.GetMem(),
		conv_desc,
		fwd_algo,
		GetNetwork().workspace,
		GetNetwork().workspace_size,
		&zero,
		y_desc,
		output.GetMem());
	if (status != CUDNN_STATUS_SUCCESS) {
		cerr << "Error: Forwarding failed in `" << name <<  "`. Error code :" << (int)status << endl;
		return false;
	}
	if (bias.GetChannels()!= 0) { 
		return output.Add(bias); 
	}
	return true;
}
bool ConvolutionalModule::Backward(FloatTensor4D & delta){
	
	if (!InferenceModule::Backward(delta)) return false;
	cudnnHandle_t handle = GetCUDNNHandle(); 

	if (bias.GetChannels() != 0) {
		cudnnConvolutionBackwardBias(handle, &one, y_desc, delta.GetMem(), &one, db_desc, dbias.GetMem());
	} 
	cudnnStatus_t status = cudnnConvolutionBackwardFilter(handle, &one, x_desc, input.GetMem(),
		y_desc, delta.GetMem(), conv_desc,bwdf_algo, GetNetwork().workspace,
		GetNetwork().workspace_size,  &one, w_desc,  dw.GetMem());
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`! Error code :" << (int)status << endl; 
		return false;
	}
	FloatTensor4D temp = delta;
	if(!delta.InitFrom(input)) return false;
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
	if (cfg.ConvParamsFreezed()) return true;
	float m = cfg.Momentum();
	float decay = (0.0f - cfg.Decay()) *  GetAppConfig().GetBatch();
	if (cfg.UpdateStrategy() == "SGD") {
		if (bias.GetChannels() != 0) { 
			if (!dbias.AddScale(bias, decay)) return false;
			if (!bias.AddScale(dbias, lr)) return false;
			// if (!dbias.Mul(m)) return false;
		}

		if (!dw.AddScale(w, decay)) return false;
		if (!w.AddScale(dw, lr)) return false;		 
		//if (!dw.Mul(m)) return false;
		
	}
	dw = 0.0f;
	dbias = 0.0f;
	return true;
}

bool ConvolutionalModule::InitDescriptors(bool trainning) {
	if (NULL == x_desc || NULL == y_desc || NULL == conv_desc || NULL == w_desc) return false;
	int b = input.GetBatch();
	int c = input.GetChannels();
	int h = input.GetHeight();
	int w = input.GetWidth();

	cudnnTensorFormat_t f = (input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	if(CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc,f, CUDNN_DATA_FLOAT, b, c, h, w))
		return false ;
	
	if (CUDNN_STATUS_SUCCESS != cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &b, &c, &h, &w))
		return false;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc,f, CUDNN_DATA_FLOAT,b,c,h,w))
		return false;
	if (!output.Init(b, c, w, h, input.GetOrder())) return false;
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

bool PoolingModule::InitDescriptors(bool trainning) {
	 
	int b = input.GetBatch();
	int c = input.GetChannels();
	int h = input.GetHeight();
	int w = input.GetWidth();
	
	
	int out_w = (w + pad_w - window_w) / stride_w + 1;
	int out_h = (h + pad_h - window_h) / stride_h + 1;
	if (!output.Init(b, c, out_w, out_h, input.GetOrder())) return false;

	if (trainning) {
		if (indexes) {
			cudaFree(indexes);
			indexes = NULL;
		}
		cudaMalloc(&indexes, output.MemElements() * sizeof(int));
	}
	
	return true;
}

PoolingModule::PoolingModule(const XMLElement * element, Layer * l, TensorOrder order, InferenceModule* prev) {
 
	x_desc = NULL;
	y_desc = NULL;
	logical_prev = prev;
	layer = l;
	input_width = 0;
	input_height = 0;
	indexes = NULL;
	const char* s = element->Attribute("type");
	string t(s? s: "max-pool"); 
	window_w = element->IntAttribute("size-w", 2);
	window_h = element->IntAttribute("size-h", 2);

	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2); 

	pad_w = element->IntAttribute("pad-w", window_w - 1);
	pad_h = element->IntAttribute("pad-h", window_h - 1);

	GetPrevModules(element);

	mode =  POOLING_MAX;
	if (t == "avg-pool") mode = POOLING_AVG; 

 
	output_channels = input_channels;
}

PoolingModule::~PoolingModule() { 
	if (indexes) cudaFree(indexes);
}
 
extern bool forward_maxpool(FloatTensor4D& output, const FloatTensor4D& input, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h);
extern bool backward_maxpool(FloatTensor4D& prev_delta, const FloatTensor4D& delta, int* indexes, int window_w, int window_h, int pad_w, int pad_h, int stride_w, int stride_h);

bool PoolingModule::Forward(ForwardContext & context) {
 
	if (!InferenceModule::Forward(context)) {
		cerr << name << " failed! \n";
		return false;
	}

	if (!forward_maxpool(output, input, indexes, window_w, window_h, pad_w, pad_h, stride_w, stride_h)) {
		cerr << name << " failed! \n";
		return false;
	} 
	return true;
}
 
bool PoolingModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;	
	if (!backward_maxpool(input,delta, indexes, window_w, window_h, pad_w, pad_h, stride_w, stride_h))
		return false; 
	delta = input; 
	return DistributeDeltas(delta);
}

BatchNormModule::BatchNormModule(const XMLElement * element, Layer * l, TensorOrder order, InferenceModule* prev) {
	layer = l;
	GetPrevModules(element);
	output_channels = input_channels;
	input_width = 0;
	input_height = 0;
	logical_prev = prev;
 
	mu = NULL;
	var = NULL;
	gamma_update = NULL;
	beta_update = NULL;
	 

	//params order :
	// beta,gamma, running_mu,running_var 
	if (params.Init(4, output_channels, 1, 1, GetNetwork().GetDataOrder())) {
		float* beta = params.GetMem();
		float* gamma = beta + output_channels;
		float* running_mu = gamma + output_channels;
		float* running_var = running_mu + output_channels;
		// bn_mean and bn_variance are for output
		mu = new_gpu_array(output_channels, 0.0);
		var = new_gpu_array(output_channels, 0.0);
		gamma_update = new_gpu_array(output_channels, 0.0);
		beta_update = new_gpu_array(output_channels, 0.0);
		cudaMemcpy(gamma, var, output_channels * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(running_var, var, output_channels * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	GetParamPool().Put(name, &params);
	 
	x_desc = NULL;
	t_desc = NULL;
	y_desc = NULL; 
	cudnnCreateTensorDescriptor(&x_desc);	
	cudnnCreateTensorDescriptor(&y_desc);	
	cudnnCreateTensorDescriptor(&t_desc);
	freezed = false; 
}

BatchNormModule::~BatchNormModule() {
	if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
	if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
	if (mu) cudaFree(mu);
	if (var) cudaFree(var);
	if (gamma_update) cudaFree(gamma_update);
	if (beta_update) cudaFree(beta_update);
}

bool BatchNormModule::InitDescriptors(bool trainning) {
	if (NULL == x_desc || NULL == y_desc || NULL == t_desc) return false;
	int b = input.GetBatch(); 
	int h = input.GetHeight();
	int w = input.GetWidth();
	cudnnTensorFormat_t f = (input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc, f, CUDNN_DATA_FLOAT,b,input_channels,h,w))
		return false;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc,f, CUDNN_DATA_FLOAT, b, output_channels, h, w))
		return false;
	cudnnStatus_t  status = cudnnDeriveBNTensorDescriptor(t_desc, x_desc, CUDNN_BATCHNORM_SPATIAL);

	return output.Init(b, output_channels, h, w, input.GetOrder());
}
 
bool BatchNormModule::Forward(ForwardContext & context) {
	 
	if (!InferenceModule::Forward(context)) return false;
	 
	float* beta = params.GetMem();
	
	float* gamma = beta + output_channels;
	float* running_mu = gamma + output_channels;
	float* running_var = running_mu + output_channels;
	cudnnStatus_t status;
	freezed = context.freezeBNParams;
	cudnnHandle_t handle = GetCUDNNHandle();
	if (context.training) {
		status = cudnnBatchNormalizationForwardTraining(handle, CUDNN_BATCHNORM_SPATIAL,
			&one, &zero, x_desc, input.GetMem(), y_desc, output.GetMem(), t_desc, gamma, beta,
			0.01f, running_mu, running_var, 1e-5, mu, var);
	}
	else {
		status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
			&one, &zero, x_desc, input.GetMem(), y_desc, output.GetMem(),
			t_desc, gamma, beta, running_mu, running_var, 1e-5);
	}
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "batch normalization failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	} 
	return true;
}

bool BatchNormModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false; 
	 
	float* beta = params.GetMem(); 
	float* gamma = beta + output_channels; 
	FloatTensor4D temp = delta;
	cudnnStatus_t status = cudnnBatchNormalizationBackward( GetCUDNNHandle(), CUDNN_BATCHNORM_SPATIAL,
		&one, &zero, &one, &one, x_desc, input.GetMem(), y_desc, temp.GetMem(), x_desc, delta.GetMem(),
		t_desc, gamma, gamma_update, beta_update, 1e-5, mu, var);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: `" << name << "` backward failed." << endl;
		return false;
	} 
	return DistributeDeltas(delta); 
}
bool BatchNormModule::UpdateParams(float lr) {
	 
	AppConfig& cfg = GetAppConfig();
	if (freezed) return true; 
	float* beta = params.GetMem();
	float* gamma = beta + output_channels;
	float* running_mu = gamma + output_channels;
	float* running_var = running_mu + output_channels;

	float decay = cfg.Decay() *  cfg.GetBatch();
	
	size_t bytes = sizeof(float) * output_channels;
	float* beta_cpu = New float[output_channels];
	cudaMemcpy(beta_cpu, beta, bytes, cudaMemcpyDeviceToHost);


	float* gamma_cpu = New float[output_channels];
	cudaMemcpy(gamma_cpu, gamma, bytes, cudaMemcpyDeviceToHost);

	float* beta_update_cpu = New float[output_channels];
	cudaMemcpy(beta_update_cpu, beta_update, bytes, cudaMemcpyDeviceToHost);


	float* gamma_update_cpu = New float[output_channels];
	cudaMemcpy(gamma_update_cpu, gamma_update, bytes, cudaMemcpyDeviceToHost);
	if (cfg.UpdateStrategy() == "SGD") {
		float m = cfg.Momentum();
		for (int i = 0; i < output_channels; i++) { 
			beta_update_cpu[i] -= decay * beta_cpu[i];
			beta_cpu[i] += lr * beta_update_cpu[i];
			beta_update_cpu[i] = 0.0f;// *= m;

			gamma_update_cpu[i] -= decay * gamma_cpu[i];
			gamma_cpu[i] += lr * gamma_update_cpu[i];
			gamma_update_cpu[i]  = 0.0f;// *= m;

		}
	}
	
	cudaMemcpy(beta, beta_cpu, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(beta_update, beta_update_cpu, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gamma, gamma_cpu, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gamma_update, gamma_update_cpu, bytes, cudaMemcpyHostToDevice);
	delete[]beta_cpu;
	delete[]gamma_cpu;
	delete[]beta_update_cpu;
	delete[]gamma_update_cpu;

	return true;
}
 
ActivationModule::ActivationModule(const XMLElement* element, Layer* l, TensorOrder order, InferenceModule* prev) {
	layer = l;
	x_desc = NULL;
	y_desc = NULL;
	logical_prev = prev;
	const char* a = element->Attribute("method");
	string str(a ? a : GetNetwork().DefaultActivation().c_str());
	if(str == "leaky") atype = LEAKY;
	else if (str == "linear") atype = LINEAR;
	else if (str == "logistic") atype = LOGISTIC;	
	else if (str == "relu") atype = RELU;
	else if (str == "lhtan") atype = LHTAN;
	else if (str == "hardtan") atype = HARDTAN;
	else if (str == "tanh") atype = TANH;
	else if (str == "loggy") atype = LOGGY;
	else if (str == "elu") atype = ELU;
	else if (str == "relie") atype = RELIE;
	else if (str == "plse") atype = PLSE;
	else if (str == "ramp") atype = RAMP;
	else if (str == "stair") atype = STAIR; 
	else atype = LINEAR;
	GetPrevModules(element);
	output_channels = input_channels;
}

bool ActivationModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	output = input;
	//output.DumpToFile(name + ".forward.before.txt");
	bool ret = activate_array_ongpu(output.GetMem(),output.MemElements(),atype);
	//output.DumpToFile(name + ".forward.after.txt");
	return ret;
}
 
bool ActivationModule::Backward(FloatTensor4D & delta) {
	
	if (!InferenceModule::Backward(delta)) return false;
	if (!output.SameDememsion(delta)) return false;
	if (!gradient_array_ongpu(output.GetMem(),delta.GetMem(),output.MemElements(), atype))
		return false ;
	return DistributeDeltas(delta);
}


UpSampleModule::UpSampleModule(const XMLElement * element, Layer * l, TensorOrder order, InferenceModule* prev) {
	layer = l;
	x_desc = NULL;
	y_desc = NULL;
	logical_prev = prev;
	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2); 
	input_width = 0;
	input_height = 0;
	GetPrevModules(element);
	output_channels = input_channels;
}

UpSampleModule::~UpSampleModule() {
}
bool UpSampleModule::InitDescriptors(bool training) {
	int b = input.GetBatch(), c = input.GetChannels(), h = input.GetHeight(), w = input.GetWidth();
	TensorOrder o = input.GetOrder();
	return output.Init(b, c, w * stride_w, h * stride_h, o); 
}
bool UpSampleModule::Forward( ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	bool ret = input.UpSample(output,stride_w,stride_h);
	return ret;
}

bool UpSampleModule::Backward(FloatTensor4D & delta) {


	if (!InferenceModule::Backward(delta)) return false;
	input = 0.0f;
	if (!delta.DownSample(input, stride_w, stride_h)) return false; 
	delta = input;  
	return DistributeDeltas(delta);
}

DeconvModule::DeconvModule(const XMLElement * element, Layer * l, TensorOrder order, InferenceModule* prev){
	input_width = 0;
	input_height = 0;
	x_desc = NULL;
	y_desc = NULL;
	logical_prev = prev;
}

DeconvModule::~DeconvModule() {
}

bool DeconvModule::Forward(ForwardContext & context) {
	//TODO: 
	return false;
}

bool DeconvModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	return DistributeDeltas(delta); 
}
