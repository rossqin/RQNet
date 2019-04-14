#include "stdafx.h"
#include "tensor.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"
#include "inference_module.h"

BatchNormModule::BatchNormModule(const XMLElement * element, Layer * l, InferenceModule* prev) : InferenceModule(element, l, prev) {
 
	

	mu = NULL;
	var = NULL;
	t_desc = NULL;
	gamma_update = NULL;
	beta_update = NULL;

	ConvolutionalModule* conv_moule = dynamic_cast<ConvolutionalModule*>(prev);
	if (NULL != conv_moule) {
		conv_moule->followed_bn_module = this;
	}	

	cudnnCreateTensorDescriptor(&x_desc);
	cudnnCreateTensorDescriptor(&y_desc);
	cudnnCreateTensorDescriptor(&t_desc);

	GetPrevModules(element);	
	InitDescriptors();
	output_channels = input_channels;
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

bool BatchNormModule::InitDescriptors() {
	if (NULL == x_desc || NULL == y_desc || NULL == t_desc) return false;
	int batch = GetAppConfig().GetMiniBatch();
	output_width = input_width;
	output_height = input_height;
	cudnnTensorFormat_t f = (input.GetOrder() == TO_NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(x_desc, f, CUDNN_DATA_FLOAT, batch, input_channels, input_height, input_width))
		return false;
	if (CUDNN_STATUS_SUCCESS != cudnnSetTensor4dDescriptor(y_desc, f, CUDNN_DATA_FLOAT, batch, output_channels, output_height, output_width))
		return false;
	return (CUDNN_STATUS_SUCCESS == cudnnDeriveBNTensorDescriptor(t_desc, x_desc, CUDNN_BATCHNORM_SPATIAL));

}

bool BatchNormModule::Forward(ForwardContext & context) {

	float one = 1.0f, zero = 0.0f;
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
	float one = 1.0f, zero = 0.0f;
	float* beta = params.GetMem();
	float* gamma = beta + output_channels;
	FloatTensor4D temp = delta;
	cudnnStatus_t status = cudnnBatchNormalizationBackward(GetCUDNNHandle(), CUDNN_BATCHNORM_SPATIAL,
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
			beta_update_cpu[i]  *= m;

			gamma_update_cpu[i] -= decay * gamma_cpu[i];
			gamma_cpu[i] += lr * gamma_update_cpu[i];
			gamma_update_cpu[i] *= m;

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
 
uint32_t BatchNormModule::GetFlops() const {
	return 0;
}
extern bool calc_weights_for_ir(float* w, const float* factors, int c_in, int size, int elements);
bool BatchNormModule::CalcWeightsForIR(FloatTensor4D& weight, FloatTensor4D& bias, float epsilon) {
	
	float* beta = reinterpret_cast<float *>(params.CopyToCPU());
	if (NULL == beta) return false; 
	float* biases_cpu = New float[output_channels];
	float* gamma = beta + output_channels;
	float* m = gamma + output_channels;
	float* v = m + output_channels;
	float* norm_gamma = New float[output_channels]; 
	for (int i = 0; i < output_channels; i++) {
		float temp = gamma[i] / sqrt(v[i] * v[i] + epsilon);
		norm_gamma[i] = temp / weight.Elements2D();
		biases_cpu[i] = beta[i] - temp;
	}
	delete[]beta;
	bias.Set3DData(0, biases_cpu, true);
	delete[]biases_cpu;

	CudaPtr<float> factors(output_channels); 
	return calc_weights_for_ir(weight.GetMem(), factors , weight.GetChannels(), weight.Elements2D(), weight.MemElements());
 
}
