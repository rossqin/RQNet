#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "inference_module.h"


ClassifierModule::ClassifierModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) :
	InferenceModule(element, l, net, prev) {
	ParsePrevModules(element);
}
bool ClassifierModule::Forward(ForwardContext& context) { 

	if (!InferenceModule::Forward(context)) return false;
	int channels = context.input->Channel();
	int batches = context.input->Batch();
	int width = context.input->Width();
	int height = context.input->Height();
	int area = width * height;
	float f = 1.0f / area;

	CpuPtr<float> input_cpu(context.input->Elements(), context.input->Data());
	CpuPtr<float> output_cpu(batches * channels);

	float* batch_data = input_cpu.ptr;
	float* output_data = output_cpu.ptr; 

	for (int b = 0; b < batches; b++) {
		float* src = batch_data;
		for (int c = 0; c < channels; c++, output_data++) {
			float sum_of_c = 0.0f;
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++, src++) {
					sum_of_c += *src;
				}
			} 
			float pred = 1.0f / (1.0f + expf(-(sum_of_c * f)));// sigmoid 
			//cout << "sum of c : " << sum_of_c << ", pred : " << pred << endl;
			*output_data = pred;
		}
		batch_data += context.input->Elements3D(); 
	}
	output.Push(output_cpu.ptr); 
	output_data = output_cpu.ptr;
	if (!context.training) { 
		for (int c = 0; c < channels; c++) {
			if (output_data[c] > GetAppConfig().ThreshHold()) {
				network->SetClassfiedResult( c , output_data[c]); 
				
			}
		} 
		 
		
		return true;
	}
	shortcut_delta.Init({ batches , channels, height, width });

	CpuPtr<float> delta_cpu(context.input->Elements());

	float* pd = delta_cpu.ptr;
	float loss = 0.0f;
	float positives = 0.0f;
	float negtives = 0.0f;
	for (int b = 0; b < batches; b++) {
		const LPObjectInfos truths = network->GetBatchTruths(b); // only one truth in the form of {0,0,0, class_id}
		if (truths->size() != 1) {
			cerr << "Error: Truth of " << network->current_training_files[b] << " expected to be 1 while " << truths->size() << endl;
			return false;
		}
		int truth_cls_id = truths->at(0).class_id;
		if (truth_cls_id < 0 || truth_cls_id >= channels) {
			cerr << "Error: Error class id " << truth_cls_id << " for " << network->current_training_files[b] << endl;
			return false;
		}
		for (int c = 0; c < channels; c++) {
			float pred = output_cpu.ptr[b * channels + c];
			float d = 0.0f;
			if (c != truth_cls_id) {// towards 0
				d = -f * focal_loss_delta(1.0f - pred);
				negtives += pred;
			}
			else {
				d = f * focal_loss_delta(pred); // towards 1 
				positives += pred;
			}
			for (int i = 0; i < area; i++) {
				pd[i] = d;
				loss += d * d * area * area ;
			}
			pd += area;
		}
	} 
	shortcut_delta.Push(delta_cpu);
	loss /= batches;
	positives /= batches;
	negtives /= (batches * (channels - 1));

	cout << fixed << setprecision(4) << "  Loss : " << loss << ", Positive :" << positives << ", Negative: " << negtives << " ...\n";
	return true;
}
bool ClassifierModule::Backward(CudaTensor& delta) {
	delta = shortcut_delta;
	return true;
}