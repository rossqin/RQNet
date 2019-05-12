#include "stdafx.h"
#include "cuda_tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"
#include <memory>

BatchNormModule::BatchNormModule(const XMLElement * element, Layer * l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) , 
	params(net->DataType(), net->DataFormat()), 
	training_params(net->DataType(), net->DataFormat()),
	adam_params(net->DataType(), net->DataFormat()) {
	t_desc = nullptr; 

	cudnnCreateTensorDescriptor(&t_desc);

	GetPrevModules(element);	

	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;
	//params order : beta, gamma, running_mu,running_var 
	params.Init(4, output_channels, 1, 1);

	CpuPtr<float> ones(output_channels);
	for (int i = 0; i < output_channels; i++)
		ones.ptr[i] = 1.0f;
	params.Push(ones.ptr, output_channels, output_channels); // set gamma to 1.0f in case we don't load the weights
	params.Push(ones.ptr, output_channels * 3, output_channels);// set running_var to 1.0f in case we don't load the weights
	
	network->weights_pool.Put(name, &params); 

	//training_params order : betas_update, gammas_update, mu, var 
	training_params.Init(4, output_channels, 1, 1);

	if (GetAppConfig().UpdatePolicy() == Adam) { 
		adam_params.Init(4, output_channels, 1, 1);
		network->adam_weights_pool.Put(name, &adam_params);
	}
	forward_input = nullptr;
	Resize(input_width, input_height);
	freezed = false;
	fused = false; 
	
}

BatchNormModule::~BatchNormModule() {
	 
}

bool BatchNormModule::Resize(int w, int h) {
	if ( nullptr == t_desc) return false; 
	input_width = w;
	input_height = h;
	output_width = input_width;
	output_height = input_height; 
	cudnnTensorDescriptor_t x_desc = input.Descriptor();
	if (!x_desc && forward_input) {
		x_desc = forward_input->Descriptor();
	}
	bool created = false;
	if (!x_desc) {
		cudnnCreateTensorDescriptor(&x_desc);
		cudnnSetTensor4dDescriptor(x_desc, input.DataFormat(), input.DataType(), network->MiniBatch(),
			input_channels, input_height, input_width);
		created = true;
	}
	bool r = (CUDNN_STATUS_SUCCESS == cudnnDeriveBNTensorDescriptor(t_desc,x_desc , CUDNN_BATCHNORM_SPATIAL));
	if (created) cudnnDestroyTensorDescriptor(x_desc);
	return r;
}

bool BatchNormModule::Forward(ForwardContext & context) {
	if (fused) {
		if (prevs.size() != 1) return false;
		context.input = &(prevs[0]->output);
		return true;
	}
	float one = 1.0f, zero = 0.0f; 
	if (!InferenceModule::Forward(context)) return false;
	if (context.input)
		forward_input = context.input;
	else
		forward_input = &input;
	void* beta = params.BatchData(0);
	void* gamma = params.BatchData(1);
	void* running_mu = params.BatchData(2);
	void* running_var = params.BatchData(3); 

	void* mu = training_params.BatchData(2);
	void* var = training_params.BatchData(3);
	cudnnStatus_t status;
	freezed = context.freezeBNParams;
	cudnnHandle_t handle = GetCUDNNHandle();
	if (context.training) { 
		status = cudnnBatchNormalizationForwardTraining(handle, CUDNN_BATCHNORM_SPATIAL,
			&one, &zero, forward_input->Descriptor(), forward_input->Data(), 
			output.Descriptor(), output, t_desc, gamma, beta,
			0.01f, running_mu, running_var, 1e-5, mu, var);
	}
	else {
		status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
			&one, &zero, forward_input->Descriptor(), forward_input->Data(), output.Descriptor(), output,
			t_desc, gamma, beta, running_mu, running_var, 1e-5);
	}
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "batch normalization failed in `" << name << "`. Error code :" << (int)status << endl;
		return false;
	}
	//input.Cache(cached_input);
	return true;
}

bool BatchNormModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	float one = 1.0f, zero = 0.0f;
	void* beta = params.BatchData(0);
	void* gamma = params.BatchData(1);
	void* beta_update = training_params.BatchData(0);
	void* gamma_update = training_params.BatchData(1);
	void* mu = training_params.BatchData(2);
	void* var = training_params.BatchData(3);
	CudaTensor temp = delta;
	//input.Restore(cached_input);
	cudnnStatus_t status = cudnnBatchNormalizationBackward(GetCUDNNHandle(), CUDNN_BATCHNORM_SPATIAL,
		&one, &zero, &one, &one, forward_input->Descriptor(), forward_input->Data(), temp.Descriptor(), temp , delta.Descriptor(), delta,
		t_desc, gamma, gamma_update, beta_update, 1e-5, mu, var);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: `" << name << "` backward failed." << endl;
		return false;
	}
	//if(input.Elements() > 0) input.Release();
	return DistributeDeltas(delta);
}
extern bool adam_update(float* theta, float* gt, float* mt, float* vt, int elements, int t, float lr, bool decay);
extern bool sgd_update(float* weights, float* updates, int elements, float lr, bool decay);
bool BatchNormModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	
	if (freezed) return true;
	int size = output_channels << 1;
	int t = network->cur_iteration;
	switch(cfg.UpdatePolicy()){
	case SGD : 
	if(params.DataType() == CUDNN_DATA_FLOAT) {
		float* weights = reinterpret_cast<float*>(params.Data());
		float* updates = reinterpret_cast<float*>(training_params.Data());
		return sgd_update(weights, updates, size, lr / cfg.GetBatch(), false);
	}
	else {
		
		CudaPtr<float> weights(size);
		CudaPtr<float> updates(size);
		__half* f16_w = reinterpret_cast<__half*>(params.Data());
		__half* f16_u = reinterpret_cast<__half*>(training_params.Data());
		if (!f16_to_f32(weights.ptr, f16_w, size)) return false;
		if (!f16_to_f32(updates.ptr, f16_u, size)) return false;
		if(!sgd_update(weights, updates, size, lr / cfg.GetBatch(), false))
			return false;

		return f32_to_f16(f16_w, weights.ptr, size);
	}
	case Adam:

		if (params.DataType() == CUDNN_DATA_FLOAT) {
			float* beta = reinterpret_cast<float*>(params.BatchData(0));
			//void* gamma = params.BatchData(1);
			float* beta_update = reinterpret_cast<float*>(training_params.BatchData(0));
			//void* gamma_update = training_params.BatchData(1);
			float* beta_m = reinterpret_cast<float*>(adam_params.BatchData(0));
			//void* gamma_m = adam_params.BatchData(1);

			float* beta_v = reinterpret_cast<float*>(adam_params.BatchData(2));
			//void* gamma_v = adam_params.BatchData(3); 

			// update beta and gamma at the same time		
			return adam_update(beta, beta_update, beta_m, beta_v, size, t,  lr, true);
		}
		else {
			CudaPtr<float> beta(size);
			CudaPtr<float> beta_update(size);
			CudaPtr<float> beta_m(size * 2); 
			float* beta_v = beta_m.ptr + size;

			__half* h_beta = reinterpret_cast<__half*>(params.BatchData(0));
			//void* gamma = params.BatchData(1);
			__half* h_beta_update = reinterpret_cast<__half*>(training_params.BatchData(0));
			//void* gamma_update = training_params.BatchData(1);
			__half* h_beta_m = reinterpret_cast<__half*>(adam_params.BatchData(0));
			//void* gamma_m = adam_params.BatchData(1);

			__half* h_beta_v = reinterpret_cast<__half*>(adam_params.BatchData(2));
			//void* gamma_v = adam_params.BatchData(3); 
			if (!f16_to_f32(beta, h_beta, size) || !f16_to_f32(beta_update, h_beta_update, size) ||
				!f16_to_f32(beta_m, h_beta_m, size * 2)) return false;
			if (!adam_update(beta, beta_update, beta_m, beta_v, size, t, lr, true)) return false;

			return f32_to_f16(h_beta, beta, size) && f32_to_f16(h_beta_m, beta_m, size * 2);
		}
		break;
	default:
		return false;
	}
	 
	return true;
}
 
uint32_t BatchNormModule::GetFlops() const {
	return 0;
}
bool fuse_batchnorm(float* filters, float* bias, float* beta, int output_channels, int filter_size);
bool BatchNormModule::Fuse() {
	if (prevs.size() != 1) return false;
	if (fused) return true;
	ConvolutionalModule* module = dynamic_cast<ConvolutionalModule*>(prevs[0]);
	if (!module) return false;  
	if (module->bias.Elements() != output_channels) {
		if (!module->bias.Init(1, output_channels, 1, 1)) return false;
	}
	bool r = false;
	if (network->DataType() == CUDNN_DATA_FLOAT) {
		float* filters = reinterpret_cast<float*>(module->w.Data());
		float* bias = reinterpret_cast<float*>(module->bias.Data());
		float* beta = reinterpret_cast<float*>(params.Data());
		r = fuse_batchnorm(filters, bias, beta, output_channels, module->w.Elements3D());
	}
	else {
		CudaPtr<float> filters(module->w.Elements());
		CudaPtr<float> bias(module->bias.Elements());
		CudaPtr<float> beta(params.Elements());
		__half* h_filters = reinterpret_cast<__half*>(module->w.Data());
		__half* h_bias = reinterpret_cast<__half*>(module->bias.Data());
		__half* h_beta = reinterpret_cast<__half*>(params.Data());

		if (!f16_to_f32(filters, h_filters, module->w.Elements()) ||
			!f16_to_f32(bias, h_bias, module->bias.Elements()) ||
			!f16_to_f32(beta, h_beta, params.Elements())) return false;

		if (!fuse_batchnorm(filters, bias, beta, output_channels, module->w.Elements3D())) return false;
		r = f32_to_f16(h_filters, filters, module->w.Elements()) &&
			f32_to_f16(h_bias, bias, module->bias.Elements());
	}
	if (r) fused = true; 
	return r;
}