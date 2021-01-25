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

	ConvolutionalModule* conv = dynamic_cast<ConvolutionalModule*>(prev);
	if (conv) conv->following_bn = this;

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

	if (GetAppConfig().PruneChannels()) {
		//TODO: check data type, presume fp32
		void* gamma = params.BatchData(1);
		void* gamma_update = training_params.BatchData(1);
		float decay = GetAppConfig().Decay(); 
		CpuPtr<float> temp(output_channels, gamma);
		CpuPtr<float> temp1(output_channels, gamma_update);
		for (int i = 0; i < output_channels; i++) {
			if (temp.ptr[i] > 0)
				temp1.ptr[i] -= decay;
			else 
				temp1.ptr[i] += decay;
		}
		training_params.Push(temp1.ptr, output_channels, output_channels);
	} //*/

	switch(cfg.UpdatePolicy()){
	case SGD : 
		return sgd_update(params.BatchData(0), training_params.BatchData(0), output_channels * 2, params.DataType(), lr , false);
 
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


bool BatchNormModule::CheckRedundantChannels(float c_threshold, float w_threshold) {
	if (prevs.size() != 1 || prevs[0].group_id != -1) return false; 

	ConvolutionalModule* module = dynamic_cast<ConvolutionalModule*>(prevs[0].module);
	if (!module) return false; 
	valid_channels = module->valid_channels;
	int prune_c = 0;
	CpuPtr<float> buffer(output_channels);
	params.Pull(buffer.ptr, output_channels, output_channels);
	for (int i = 0; i < output_channels; i++) {
		if (!valid_channels[i]) continue;
		if (fabs(buffer.ptr[i]) < c_threshold) {
			cout << "  * channel " << i << ": gamma = " << buffer.ptr[i] << "\n";
			valid_channels[i] = false;
			prune_c++;
		}
	}
	if (prune_c == 0) return true; 
	prune_c = 0;
	for (int i = 0; i < output_channels; i++) {
		module->valid_channels[i] = valid_channels[i];
		if (!valid_channels[i]) prune_c++; 
	}
	cout << " Redundant Channels in " << module->Name() << " : " << prune_c << ".\n\n";
	return true;
}
bool BatchNormModule::Prune() { 
	valid_channels = prevs[0].module->valid_channels;
	int new_oc = 0;
	for (int i = 0; i < output_channels; i++) {
		if (valid_channels[i]) new_oc++;
	}
	if (new_oc == output_channels) return true;
	CpuPtr<float> buffer(params.Elements());
	params.Pull(buffer.ptr, 0, params.Elements());
	int dest = 0;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < output_channels; j++) {
			if (valid_channels[j]) {
				buffer.ptr[dest++] = buffer[i * output_channels + j];
			}
		}
	}
	if (!params.Init({ 4,new_oc,1,1 })) return false;
	if (!params.Push(buffer.ptr, 0, dest)) return false;
	
	cout << " Channels of " << name << " reduced from " << output_channels << " to " << new_oc << ".\n\n";
	output_channels = new_oc;

	return true;
}