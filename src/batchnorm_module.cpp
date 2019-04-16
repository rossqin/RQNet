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
	training_params(net->DataType(), net->DataFormat()){
	t_desc = nullptr; 

	cudnnCreateTensorDescriptor(&t_desc);

	GetPrevModules(element);	

	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;
	//params order :
	// beta,gamma, running_mu,running_var 
	params.Init(4, output_channels, 1, 1);
	
	
	if (net->DataFormat() == CUDNN_DATA_FLOAT) {
		float* buffer = New float[output_channels];
		for (int i = 0; i < output_channels; i++)
			buffer[i] = 1.0f;
		void* gamma = params.BatchData(1);
		cudaMemcpy(gamma, buffer, output_channels * sizeof(float), cudaMemcpyHostToDevice);
		delete[]buffer;
	}
	else {
		__half* buffer = New __half[output_channels];
		for (int i = 0; i < output_channels; i++)
			buffer[i] = __float2half(1.0f);
		void* gamma = params.BatchData(1);
		cudaMemcpy(gamma, buffer, output_channels * sizeof(__half), cudaMemcpyHostToDevice);
		delete[]buffer;
	}
	training_params.Init(4, output_channels, 1, 1);
	GetParamPool().Put(name, &params); 
	
	freezed = false;
	fused = false;
}

BatchNormModule::~BatchNormModule() {
	 
}

bool BatchNormModule::Resize(int w, int h) {
	if ( NULL == t_desc) return false; 
	input_width = w;
	input_height = h;
	output_width = input_width;
	output_height = input_height;
	return (CUDNN_STATUS_SUCCESS == cudnnDeriveBNTensorDescriptor(t_desc, input.Descriptor(), CUDNN_BATCHNORM_SPATIAL));

}

bool BatchNormModule::Forward(ForwardContext & context) {
	if (fused) return true;
	float one = 1.0f, zero = 0.0f;
	if (!InferenceModule::Forward(context)) return false;

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
			&one, &zero, input.Descriptor(), input, output.Descriptor(), output, t_desc, gamma, beta,
			0.01f, running_mu, running_var, 1e-5, mu, var);
	}
	else {
		status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
			&one, &zero, input.Descriptor(), input, output.Descriptor(), output,
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
		&one, &zero, &one, &one, input.Descriptor(), input, temp.Descriptor(), temp , delta.Descriptor(), delta,
		t_desc, gamma, gamma_update, beta_update, 1e-5, mu, var);
	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << "Error: `" << name << "` backward failed." << endl;
		return false;
	}
	return DistributeDeltas(delta);
}
extern bool sgd_update(void* params, void* updates, int elements, cudnnDataType_t data_type, float lr, float decay, float momentum);
bool BatchNormModule::UpdateParams(float lr) {

	AppConfig& cfg = GetAppConfig();
	
	if (freezed) return true;
	
	if (cfg.UpdateStrategy() == "SGD") {
		return sgd_update(params, training_params, output_channels * 2, params.DataType(), lr / cfg.GetBatch(), cfg.Decay() / cfg.GetBatch(), cfg.Momentum());
	}
	 
	return false;
}
 
uint32_t BatchNormModule::GetFlops() const {
	return 0;
}
bool fuse_batchnorm(void* filters, void* bias, void* batchnorm_params, int channels, int w, int h, int channels_in, cudnnDataType_t data_type);
bool BatchNormModule::Fuse() {
	if (prevs.size() != 1) return false;
	if (fused) return true;
	ConvolutionalModule* module = dynamic_cast<ConvolutionalModule*>(prevs[0]);
	if (!module) return false; 
	void* beta = params.BatchData(0);
	void* gamma = params.BatchData(1);
	if (module->bias.Elements() != output_channels) {
		if (module->bias.Init(1, output_channels, 1, 1)) return false;
	}
	bool r = fuse_batchnorm(module->w, module->bias,params,output_channels, module->w.Width(), module->w.Height(), module->input_channels, module->w.DataType());
	if (r) {
		fused = true;
	}
	return r;
}