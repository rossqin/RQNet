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

	ParsePrevModules(element);	

	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;
	//params order : beta, gamma, running_mu,running_var 
	params.Init({ 4, output_channels, 1, 1 });

	CpuPtr<float> ones(output_channels);
	for (int i = 0; i < output_channels; i++)
		ones.ptr[i] = 1.0f;
	params.Push(ones.ptr, output_channels, output_channels); // set gamma to 1.0f in case we don't load the weights
	params.Push(ones.ptr, output_channels * 3, output_channels);// set running_var to 1.0f in case we don't load the weights
	
	network->weights_pool.Put(name, &params); 

	//training_params order : betas_update, gammas_update, mu, var 
	training_params.Init({ 4, output_channels, 1, 1 });

	if (GetAppConfig().UpdatePolicy() == Adam) { 
		adam_params.Init({ 4, output_channels, 1, 1 });
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
	
	float one = 1.0f, zero = 0.0f; 
	if (!InferenceModule::Forward(context)) return false;
	forward_input = context.input;
	if (fused) {
		output = *(context.input);
		return true;
	}

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
	cudnnStatus_t status = cudnnBatchNormalizationBackward(GetCUDNNHandle(), CUDNN_BATCHNORM_SPATIAL,
		&one, &zero, &one, &one, forward_input->Descriptor(), forward_input->Data(), temp.Descriptor(), temp , delta.Descriptor(), delta,
		t_desc, gamma, gamma_update, beta_update, 1e-5, mu, var);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: `" << name << "` backward failed." << endl;
		return false;
	} 
	return DistributeDeltas(delta);
}
extern bool adam_update(void* theta, void* gt, void* mt, void* vt, int elements, int t, cudnnDataType_t data_type, float lr, bool decay);
extern bool sgd_update(void* weights, void* updates, int elements, cudnnDataType_t data_type, float lr, bool decay);
bool BatchNormModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	
	if (freezed) return true;

	switch(cfg.UpdatePolicy()){
	case SGD : 
		return sgd_update(params, training_params, output_channels * 2, params.DataType(), lr , false);
 
	case Adam:
		{
		void* beta = params.BatchData(0);
		//void* gamma = params.BatchData(1);
		void* beta_update = training_params.BatchData(0);
		//void* gamma_update = training_params.BatchData(1);
		void* beta_m = adam_params.BatchData(0);
		//void* gamma_m = adam_params.BatchData(1);

		void* beta_v = adam_params.BatchData(2);
		//void* gamma_v = adam_params.BatchData(3); 

		// update beta and gamma at the same time
		int t = network->cur_iteration;
		return adam_update(beta, beta_update, beta_m, beta_v, 2 * output_channels, t, params.DataType(), lr, false);
		}
	default:
		return true;
	}
	 
	return true;
}
 
uint32_t BatchNormModule::GetFlops() const {
	return 0;
}
bool fuse_batchnorm(void* filters, void* bias, void* batchnorm_params, 
	int output_channels, int filter_size, cudnnDataType_t data_type);
bool BatchNormModule::Fuse() {
	if (prevs.size() != 1) return false;
	if (fused) return true;
	ConvolutionalModule* module = dynamic_cast<ConvolutionalModule*>(prevs[0].module);
	if (!module) return false;  
	if (module->bias.Elements() != output_channels) {
		if (!module->bias.Init({ 1, output_channels, 1, 1 })) return false;
	}
	bool r = fuse_batchnorm(module->w, module->bias,params,output_channels,
		module->w.Elements3D(), module->w.DataType());
	if (r) fused = true; 
	return r;
}