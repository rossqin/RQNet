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
 
InferenceModule* InferenceModule::FromXmlElement(const XMLElement* element,  Layer* layer, TensorOrder order) {
	const char *t = element->Attribute("type");
	string mtype(t ? t : "");
	InferenceModule* module = NULL;
	if (mtype == "conv" || mtype == "convolutional") {
		module = New ConvolutionalModule(element, layer, order);
	}
	else if (mtype == "batch-norm" ) {
		module = New BatchNormModule(element, layer, order);
	}
	else if(mtype == "activation") {
		module = New ActivationModule(element, layer, order);
	}
	else if (mtype == "max-pool" || mtype == "avg-pool") {
		module = New PoolingModule(element, layer, order);
	}
	else if (mtype == "upsample") {
		module = New UpSampleModule(element, layer, order);
	}
	else if (mtype == "yolo-detection") {
		module = New YoloModule(element, layer, order);
	}
	//TODO: Add your New Moule here.

	if (module)
		GetModulePool().insert(pair<string, InferenceModule*>(module->name, module));
	return module;
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
		if (!input.Init(GetAppConfig().GetMiniBatch(), input_channels,
			prevs[0]->output.GetWidth(), prevs[0]->output.GetHeight(), prevs[0]->output.GetOrder()))
			return false;
		
		float* dest = input.GetMem();
		 
		size_t offset = 0;
		for (int i = 0; i < n; i++  ) {
			InferenceModule* module = prevs[i];
			float* src = module->output.GetMem(); 
			size_t elements = module->output.Elements3D();
			float *temp = dest;
			for (int j = 0; j < input.GetBatch(); j++ , src += elements , temp += input.Elements3D() ) {				
				cudaError_t err = cudaMemcpy(reinterpret_cast<char*>(temp),
					reinterpret_cast<char*>(src), elements * sizeof(float) , cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) {
					cerr << "Error: cudaMemcpy failed in `"<<name<<"`, batch : " << j << "! " << endl;
					return false;
				}
			}
			dest += elements;
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
	if (delta.SameDememsion(shortcut_delta)) {
		return delta.Add(shortcut_delta);
	}
	return true;
}
bool InferenceModule::DistributeDeltas(FloatTensor4D & delta) {
	int n = prevs.size();
	if (n < 2) return true;
	 
	 
	float* d = delta.GetMem();
	float* d_end = d + delta.MemElements();
	for (int i = 0; i < n; i++) {
		InferenceModule* module =prevs[i];
		unsigned char* input_gpu = NULL; 
		FloatTensor4D& sd = module->shortcut_delta;
		FloatTensor4D& o = module->output;
		if((!sd.SameDememsion(o)) &&  
			(!sd.Init(o.GetBatch(),o.GetChannels(),o.GetWidth(),o.GetHeight(),o.GetOrder()))) {
			cerr << "Error: DistributeDeltas failed in `"<<name <<"` !\n "  ;
			return false;
		}
		float *dst = sd.GetMem();
		float *dst_end = dst + sd.MemElements();
		for (int b = 0; b < delta.GetBatch(); b++ , dst += sd.Elements3D()) {
			float* src = d + b * delta.Elements3D();
			if (dst > dst_end || src > d_end) {
				cout << "out of bound!\n";
				break;
			}
			if (!add_in_gpu(dst, src, sd.Elements3D())) {
				return false;
			}			
		} 
		d += sd.Elements3D();
	}
	delta.Release();
	FloatTensor4D& o = prevs[0]->output;
	return delta.Init(o.GetBatch(), o.GetChannels(), o.GetWidth(), o.GetHeight(), o.GetOrder());
}
ConvolutionalModule::~ConvolutionalModule() {
	if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
	if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
	if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
	if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
	if (db_desc) cudnnDestroyTensorDescriptor(db_desc);
}

ConvolutionalModule::ConvolutionalModule(const XMLElement * element, Layer * l, TensorOrder order)  {
	input = NULL;
	db_desc = NULL;
	layer = l;
	input_width = 0;
	input_height = 0;
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
	cudnnStatus_t status;
	if (bias.GetChannels() != 0) {
		status = cudnnConvolutionBackwardBias(handle, &one, y_desc, delta.GetMem(), &one, db_desc, dbias.GetMem());
		if (CUDNN_STATUS_SUCCESS != status) {
			cerr << "Error: backward bias failed in`" << name << "`. Error code :" << (int)status << endl;
			return false;
		}
	}
	status = cudnnConvolutionBackwardFilter(handle,
		&one, x_desc, 
		input.GetMem(),
		y_desc, 
		delta.GetMem(), 
		conv_desc,
		bwdf_algo, 
		GetNetwork().workspace,
		GetNetwork().workspace_size, 
		&one,
		w_desc, 
		dw.GetMem());
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: backward filter failed in`" << name << "`. Error code :" << (int)status << endl; 
		return false;
	}
	
	if (prevs.size() > 0) {
		status = cudnnConvolutionBackwardData(handle,
			&one, 
			w_desc, 
			w.GetMem(),
			y_desc, 
			delta.GetMem(), 
			conv_desc, 
			bwdd_algo, 
			GetNetwork().workspace, 
			GetNetwork().workspace_size, 
			&zero,
			x_desc, 
			input.GetMem());
		if (CUDNN_STATUS_SUCCESS != status) {
			cerr << "Error: backward data failed in`" << name << "`. Error code :" << (int)status << endl;
			return false;
		} //*/ 
		delta = input;
	}
	return DistributeDeltas(delta); 
}

bool ConvolutionalModule::UpdateParams(float lr) {
	AppConfig& cfg = GetAppConfig(); 
	if (cfg.ConvParamsFreezed()) return true;
	float m = cfg.Momentum();
	float decay = 0.0f - cfg.Decay();
	if (cfg.UpdateStrategy() == "SGD") {

		if (bias.GetChannels() != 0) { 
			//if (!dbias.AddScale(bias,-cfg.Decay())) return false;
			if (!bias.AddScale(dbias, lr)) return false;
			 if (!dbias.Mul(m)) return false;
		}

		if (!dw.AddScale(w, decay)) return false;
		if (!w.AddScale(dw, lr)) return false;		 
		if (!dw.Mul(m)) return false;
		
	}
	return true;
}

bool ConvolutionalModule::InitDescriptors(bool trainning) {
	if (NULL == x_desc || NULL == y_desc || NULL == conv_desc || NULL == w_desc) return false;
	if(CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc,
		(input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,
		input.GetBatch(),
		input.GetChannels(),
		input.GetHeight(),
		input.GetWidth()))
		return false ;
	int b = 0, c = 0, h = 0, w = 0;
	if (CUDNN_STATUS_SUCCESS != cudnnGetConvolution2dForwardOutputDim(conv_desc,
		x_desc, w_desc, &b, &c, &h, &w))
		return false;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc,
		(input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
		CUDNN_DATA_FLOAT,b,c,h,w))
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
	 
	int b = input.GetBatch(), c = input.GetChannels(), h = input.GetHeight(), w = input.GetWidth();
	
	if (stride_w != 1 && stride_h != 1) {
		if (NULL == x_desc || NULL == y_desc || NULL == desc) return false;
		if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc,
			(input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
			CUDNN_DATA_FLOAT,b, c, h, w	))
			return false;
		if (CUDNN_STATUS_SUCCESS != cudnnGetPooling2dForwardOutputDim(desc, x_desc, &b, &c, &h, &w))
			return false;
		if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc,
			(input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC,
			CUDNN_DATA_FLOAT, b, c, h, w))
			return false;
		
	}
	else {
		if (indexes) {
			cudaFree(indexes);
			indexes = NULL;
		}
		cudaMalloc(&indexes, input.MemElements() * sizeof(int));
	}
	if (!output.Init(b, c, w, h, input.GetOrder())) return false;
	
	return true;
}

PoolingModule::PoolingModule(const XMLElement * element, Layer * l, TensorOrder order) {
	desc = NULL;
	x_desc = NULL;
	y_desc = NULL;
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

	GetPrevModules(element);

	mode = CUDNN_POOLING_MAX;
	if (t == "max-pool")
		mode = CUDNN_POOLING_MAX;
	else if (t == "avg-pool")
		mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	else if (t == "avg-pool-no-padding")
		mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
	else if (t == "avg-pool-deterministic")
		mode = CUDNN_POOLING_MAX_DETERMINISTIC;	
	if (stride_w != 1 && stride_h != 1) {
		cudnnCreatePoolingDescriptor(&desc);
		if (desc) {
			if (CUDNN_STATUS_SUCCESS != cudnnSetPooling2dDescriptor(desc, mode,
				CUDNN_NOT_PROPAGATE_NAN, window_h, window_w, 0, 0, stride_h, stride_w)) {
				cudnnDestroyPoolingDescriptor(desc);
				desc = NULL;
			}
		}
		cudnnCreateTensorDescriptor(&x_desc);
		cudnnCreateTensorDescriptor(&y_desc);
	}
	output_channels = input_channels;
}

PoolingModule::~PoolingModule() {
	if(desc) cudnnDestroyPoolingDescriptor(desc);
	if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
	if (y_desc) cudnnDestroyTensorDescriptor(y_desc); 
	if (indexes) cudaFree(indexes);
}
extern bool forward_one_stride_maxpool(FloatTensor4D& output, const FloatTensor4D& input, int* indexes, int window_w, int window_h);
void dump_gpu_data(const string& filename ,int* data, int width, int height) {
	int s = width * height;
	int* buffer = new int[s];
	memset(buffer, 0, sizeof(int) * s);
	 
	cudaMemcpy(buffer, data , s * sizeof(int), cudaMemcpyDeviceToHost);
	 
	ofstream f(filename, ios::trunc);
	if (f.is_open()) {
		char temp[20]; 
		int i = 0;
		for (int y = 0; y < height; y++) {
			f << endl;
			for (int x = 0; x < width; x++, i++) {
				sprintf(temp, "%d ", buffer[i]);
				f << setw(6) << temp;
			}
		}

		f.close();
	}
	delete[]buffer;
}
bool PoolingModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	//fix: stride == 1 
	 	 
		
	if (stride_w != 1 && stride_h != 1) {
		if (CUDNN_STATUS_SUCCESS != cudnnPoolingForward(GetCUDNNHandle(),
			desc, &one, x_desc, input.GetMem(), &zero, y_desc, output.GetMem()))
			return false;
	}
	else {
		input.DumpToFile(name + ".forward.01.txt",2,2);
		if(!forward_one_stride_maxpool(output, input, indexes, window_w, window_h))
			return false; 
		output.DumpToFile(name + ".forward.02.txt", 2, 2);
		int* temp = indexes + (2 * input_channels + 2) * input_height * input_width;
		dump_gpu_data(name + ".forward.indexes.txt", temp, input_width, input_height);
		//return false;
	}
	//
	//
	return true;
}
extern bool backward_one_stride_maxpool(FloatTensor4D& delta, int* indexes);
bool PoolingModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	//
	if (stride_w != 1 && stride_h != 1) {
		FloatTensor4D dy(delta);
		delta.Release();
		if (!delta.Init(input.GetBatch(), input.GetChannels(), input.GetWidth(),
			input.GetHeight(), input.GetOrder()))
			return false;
		if (CUDNN_STATUS_SUCCESS != cudnnPoolingBackward(GetCUDNNHandle(),
			desc, &one, y_desc, output.GetMem(), y_desc, dy.GetMem(),
			x_desc, input.GetMem(), &zero, x_desc, delta.GetMem()))
			return false;
	}
	else {
		delta.DumpToFile(name + ".backward.01.txt",2,2);
		if(!backward_one_stride_maxpool(delta, indexes))
			return false;
		delta.DumpToFile(name + ".backward.02.txt",2,2);
	}
	
	return DistributeDeltas(delta);
}

BatchNormModule::BatchNormModule(const XMLElement * element, Layer * l, TensorOrder order) {
	layer = l;
	GetPrevModules(element);
	output_channels = input_channels;
	input_width = 0;
	input_height = 0;
 
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
		var = new_gpu_array(output_channels, 1.0);
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
	/*if (t_desc) {
		if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(t_desc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, input_channels, 1, 1)) {
			cudnnDestroyTensorDescriptor(t_desc);
			t_desc = NULL;
		}
	}*/
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
	cudnnTensorFormat_t o = (input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc,o, CUDNN_DATA_FLOAT,b,input_channels,h,w))
		return false;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc, o, CUDNN_DATA_FLOAT, b, output_channels, h, w))
		return false;
	cudnnStatus_t  status = cudnnDeriveBNTensorDescriptor(t_desc, x_desc, CUDNN_BATCHNORM_SPATIAL);

	return output.Init(b, output_channels, h, w, input.GetOrder());
}
const float AVERAGE_MOVING_FACTOR = 0.01f;
const float BN_MIN_EPSILON = 1.0e-5;
bool BatchNormModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	 
	float* beta = params.RestoreDataFromCPU();
	float* gamma = beta + output_channels;
	float* running_mu = gamma + output_channels;
	float* running_var = running_mu + output_channels;
	cudnnStatus_t status;
	freezed = context.freezeBNParams;
	if (context.training) {
		status = cudnnBatchNormalizationForwardTraining(
			GetCUDNNHandle(),
			CUDNN_BATCHNORM_SPATIAL,
			&one,
			&zero,
			x_desc,
			input.GetMem(),
			y_desc,
			output.GetMem(),
			t_desc,
			gamma,
			beta,
			AVERAGE_MOVING_FACTOR,
			running_mu,
			running_var,
			BN_MIN_EPSILON,
			mu,
			var);
		//float* output_cpu = New float[output.MemElements()];
		//cudaMemcpy(output_cpu, output.GetMem(), output.MemBytes() ,cudaMemcpyDeviceToHost);
		//dump_mem(output_cpu, 100);

		//delete[]output_cpu;
	}
	else {
		status = cudnnBatchNormalizationForwardInference(
			GetCUDNNHandle(),
			CUDNN_BATCHNORM_SPATIAL,
			&one,
			&zero,
			x_desc,
			input.GetMem(),
			y_desc,
			output.GetMem(),
			t_desc,
			gamma,
			beta,
			running_mu,
			running_var,
			BN_MIN_EPSILON);
	}
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "batch normalization failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	} 
	return true;
}

bool BatchNormModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	// this function will not be invoked during testing phase
	FloatTensor4D dx;
	if (!dx.Init(delta.GetBatch(),
		delta.GetChannels(),
		delta.GetHeight(),
		delta.GetHeight(),delta.GetOrder() ) ) return false; 
	float* beta = params.GetMem(); 
	float* gamma = beta + output_channels;
	//float* running_mu = gamma + output_channels;
	//float* running_var = running_mu + output_channels;
	cudnnStatus_t status = cudnnBatchNormalizationBackward(
		GetCUDNNHandle(),
		CUDNN_BATCHNORM_SPATIAL,
		&one,
		&zero,
		&one,
		&one,
		x_desc,
		input.GetMem(),
		y_desc,
		delta.GetMem(),
		x_desc,
		dx.GetMem(),
		t_desc,
		gamma,
		gamma_update,
		beta_update,
		BN_MIN_EPSILON,
		 mu,
		var);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: Normalizer back warding failed in `"
			<< name << "` ,BatchNormalization failed." << endl;
		return false;
	}
	delta = dx; 
	return DistributeDeltas(delta); 
}
bool BatchNormModule::UpdateParams(float lr) {
	AppConfig& cfg = GetAppConfig();
	if (freezed) return true; 
	float* beta = params.GetMem();
	float* gamma = beta + output_channels;
	float* running_mu = gamma + output_channels;
	float* running_var = running_mu + output_channels;

	/*
	float* mu;
	float* var;
	float* gamma_update;
	float* beta_update;
	*/
	float decay = cfg.Decay();
	
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
			beta_update_cpu[i] *= m;

			gamma_update_cpu[i] -= decay * gamma_cpu[i];
			gamma_cpu[i] += lr * gamma_update_cpu[i];
			gamma_update_cpu[i] = m;

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
 
ActivationModule::ActivationModule(const XMLElement* element, Layer* l, TensorOrder order) {
	layer = l;
	x_desc = NULL;
	y_desc = NULL;
	
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
	//delta.DumpToFile(name + ".backward.before.txt");
	if (!gradient_array_ongpu(output.GetMem(),delta.GetMem(),output.MemElements(), atype))
		return false ;
	//delta.DumpToFile(name + ".backward.after.txt");
	return DistributeDeltas(delta);
}


UpSampleModule::UpSampleModule(const XMLElement * element, Layer * l, TensorOrder order) {
	layer = l;
	x_desc = NULL;
	y_desc = NULL;
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
	//input.DumpToFile(name + ".forward.before.txt");
	bool ret = input.UpSample(output,stride_w,stride_h);
	//output.DumpToFile(name + ".forward.after.txt");
	return ret;
}

bool UpSampleModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	//delta.DumpToFile(name + ".backward.01.txt");
	if(!delta.DownSample(input,stride_w,stride_h)) return false;
	delta = input; 
	//input.DumpToFile(name + ".backward.02.txt");
	//delta.DumpToFile(name + ".backward.03.txt");
	return DistributeDeltas(delta);
}

DeconvModule::DeconvModule(const XMLElement * element, Layer * l, TensorOrder order)
{
	input_width = 0;
	input_height = 0;
}

DeconvModule::~DeconvModule()
{
}

bool DeconvModule::Forward(ForwardContext & context)
{

	 
	return false;
}

bool DeconvModule::Backward(FloatTensor4D & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	return DistributeDeltas(delta); 
}
