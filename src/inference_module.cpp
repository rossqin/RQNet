#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "cuda_tensor.h"
#include "yolo.h"
#include "inference_module.h"
#include "config.h"
const float one = 1.0f, zero = 0.0f;
 
InferenceModule::InferenceModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) 
	: input(net->DataType(), net->DataFormat()), output(net->DataType(), net->DataFormat()), 
	shortcut_delta(net->DataType(), net->DataFormat()) {
	if (prev)
		index = prev->index + 1;
	else
		index = 1;
	network = net;
	output_port = 3;
	layer = l;
	input_channels = 0;
	input_height = 0;
	input_width = 0;
	output_channels = 0;
	output_height = 0;
	output_width = 0;
	logical_prev = prev;
	string base = layer->GetName() + ".";
	const char* id = element->Attribute("id");
	if (NULL == id) id = "convolution";
	name = base + id;
	cout << " INFO: Initializing " << name << " ...\n";
}
void InferenceModule::GetPrevModules(const XMLElement* element) {
	string base = layer->GetName() + ".";
	ModulePool& module_pool = GetModulePool();	
	const char *p = element->Attribute("before");	 
	string prev_ids_s(p ? p : "none");
	input_channels = 0;
		 
	if (prev_ids_s.length() > 0 && prev_ids_s != "none") {
		vector<string> prev_ids;
		ModulePool::iterator  it;
		split_string(prev_ids, prev_ids_s);		
		for (auto id : prev_ids) {
			if (id.find('.') == string::npos)
				it = module_pool.find(base + id);
			else
				it = module_pool.find(id);
			if (it != module_pool.end()) {
				InferenceModule* module = it->second ;
				input_channels += module->output_channels;
				if (input_width < module->output_width) input_width = module->output_width;
				if (input_height <module->output_height) input_height = module->output_height;
				prevs.push_back(it->second);
				
			}
		}
	}
	else {
		input_channels = network->input_channels;
		input_width = network->input_width;
		input_height = network->input_height;
	}
	
	if (prevs.size() > 1) {
		index ++; // save for later IR transform
	}
}
 
InferenceModule* InferenceModule::FromXmlElement(const XMLElement* element,  Layer* layer, CNNNetwork* network, InferenceModule* prev) {
	const char *t = element->Attribute("type");
	string mtype(t ? t : "");
	InferenceModule* module = NULL;
	if (mtype == "conv" || mtype == "convolutional") {
		module = New ConvolutionalModule(element, layer, network, prev);
	}
	else if (mtype == "batch-norm" ) {
		module = New BatchNormModule(element, layer, network, prev);
	}
	else if(mtype == "activation") {
		module = New ActivationModule(element, layer, network, prev);
	}
	else if (mtype == "max-pool" || mtype == "avg-pool") {
		module = New PoolingModule(element, layer, network, prev);
	}
	else if (mtype == "upsample") {
		module = New UpSampleModule(element, layer, network, prev);
	}
	else if (mtype == "yolo-detection") {
		module = New YoloModule(element, layer, network, prev);
	}
	//TODO: Add your New Moule here.

	if (module)
		GetModulePool().insert(pair<string, InferenceModule*>(module->name, module));
	return module;
}
void InferenceModule::WritePorts(ofstream& xml) const {
	xml << "      <input>" << endl;
	xml << "        <port id=\"0\">" << endl;
	xml << "          <dim>1</dim>" << endl;
	xml << "          <dim>" << input_channels << "</dim>" << endl;
	xml << "          <dim>" << input_height << "</dim>" << endl;
	xml << "          <dim>" << input_width << "</dim>" << endl;
	xml << "        </port>" << endl;
	xml << "      </input>" << endl;
	xml << "      <output>" << endl;
	// not know why id==3
	xml << "        <port id=\"" << output_port << "\">" << endl;
	xml << "          <dim>1</dim>" << endl;
	xml << "          <dim>" << output_channels << "</dim>" << endl;
	xml << "          <dim>" << output_height << "</dim>" << endl;
	xml << "          <dim>" << output_width << "</dim>" << endl;
	xml << "        </port>" << endl;
	xml << "      </output>" << endl;
}
bool InferenceModule::PrepareShortcutDelta() {
	if (shortcut_delta.SameShape(output))
		return true;
	return shortcut_delta.Init(input.Batch(), input.Channel(), input.Height(), input.Width());
}
bool InferenceModule::UpdateShortcutDelta(const CudaTensor& delta) {
	if (shortcut_delta.Elements() == 0) {
		shortcut_delta = delta;
		return true;
	}
	if (shortcut_delta.SameShape(delta)) {
		return shortcut_delta.Add(delta);
	}
	return false;
}
bool InferenceModule::Forward(ForwardContext & context) {
	int n = (int)prevs.size();  
	int w = 0;
	int h = 0;

	if (n == 1) { 
		input = prevs[0]->output;
		w = input.Width();
		h = input.Height();
	}
	else if (n > 1) {
		input = 0.0f;		
		vector<const CudaTensor*> srcs;
		for (int i = 0; i < n; i++) {
			if (prevs[i]->output.Height() > h)
				h = prevs[i]->output.Height();
			if (prevs[i]->output.Width() > w)
				w = prevs[i]->output.Width();
			srcs.push_back(&(prevs[i]->output));
		} 
		int expected_elements = network->mini_batch * network->input_channels * w * h;
		if (w != input_width || h != input_height || input.Elements() != expected_elements) {
			if (!input.Init(network->mini_batch, input_channels, w, h)) return false;
		}
		else {
			input = 0.0f;
		}
		if (!input.Concat(srcs)) return false;
	}
	else {
		//input = context.input;
		w = network->input_width;
		h = network->input_height;
		int expected_elements = network->mini_batch * network->input_channels * w * h;
		if (input_width != w || input_height != h || input.Elements() != expected_elements) {
			if (!input.Init(network->mini_batch, input_channels, w, h)) return false;
			if(!input.Push(network->input)) return false;
		}
	}
	if (input_width != w || input_height != h) {
		if (!Resize(w, h)) return false; 
			
	}
	int expected_output_elemens = network->mini_batch * output_channels * output_height * output_width;
	if (output.Elements() != expected_output_elemens &&
		!output.Init(network->mini_batch, output_channels, output_height, output_width)) return false;
	context.input = &output;
	return true;
}
 
bool InferenceModule::Backward(CudaTensor & delta) {
	if (delta.Elements() == 0 && shortcut_delta.Elements() != 0) {
		delta = shortcut_delta;
		return shortcut_delta.Release();
	}
	if (delta.SameShape(shortcut_delta)) {
		if(!delta.Add(shortcut_delta)) return false;
		return shortcut_delta.Release();
	}
	return true;
}
bool InferenceModule::DistributeDeltas(CudaTensor & delta) {
	int n = prevs.size();
	if (n == 0) return true;
	if (n < 2) {
		InferenceModule* module = prevs[0];
		if (module != logical_prev) {			 
			return module->shortcut_delta.Add(delta);
		}
		return true;
	}
	vector<CudaTensor*> prev_deltas;
	for (int i = 0, c = 0; i < n; i++) {
		InferenceModule* module = prevs[i]; 
		if (!module->PrepareShortcutDelta()) return false;
		prev_deltas.push_back(&(module->shortcut_delta)); 
	} 
	if (!delta.Split(prev_deltas))
		return false;
	return delta.Release(); 
}

bool InferenceModule::OutputIRModel(ofstream & xml, ofstream & bin, stringstream & edges, size_t & bin_offset, bool fp16) const {
	int n = prevs.size();
	if (0 == n) {
		edges << "    <edge from-layer=\"0\" from-port=\"0\" to-layer=\"" << index 
			<< "\" to-port=\"0\"/>" << endl;
	}
	int output_index = index;
	if (n > 1) output_index--;
	for (int i = 0; i < n; i++) {
		InferenceModule* module = prevs[i];
		edges << "    <edge from-layer=\"" << module->index << "\" from-port=\"" << module->output_port <<
			"\" to-layer=\"" << output_index << "\" to-port=\"" << i << "\"/>" << endl;
	}
	if (n < 2) return true;
	string concat_name = name + ".concat";
	xml << "    <layer id=\"" << output_index << "\" name=\"" << concat_name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"Concat\">" << endl;
	xml << "      <data axis=\"1\"/>" << endl;
	xml << "      <input>" << endl;
	for (int i = 0; i < n; i++) {
		InferenceModule* module = prevs[i];
		xml << "        <port id=\""<<i<<"\">" << endl;
		xml << "          <dim>1</dim>" << endl;
		xml << "          <dim>" << module->output_channels << "</dim>" << endl;
		xml << "          <dim>" << module->output_height << "</dim>" << endl;
		xml << "          <dim>" << module->output_width << "</dim>" << endl;
		xml << "        </port>" << endl;
	}
	xml << "      </input>" << endl;
	xml << "      <output id=\""<<n<<"\">" << endl;
	xml << "          <dim>1</dim>" << endl;
	xml << "          <dim>" << input_channels << "</dim>" << endl;
	xml << "          <dim>" << input_height << "</dim>" << endl;
	xml << "          <dim>" << input_width << "</dim>" << endl;
	xml << "      </output>" << endl;
	edges << "    <edge from-layer=\"" << output_index << "\" from-port=\"" << n <<
		"\" to-layer=\"" << index << "\" to-port=\"0\"/>" << endl;
	return true;
}
 
bool ActivationModule::Resize(int w, int h) { 
	input_height = h;
	input_width = w;
	output_width = input_width; 
	output_height = input_height; 
	
	return true;
}

ActivationModule::ActivationModule(const XMLElement* element, Layer* l,CNNNetwork* net, InferenceModule* prev):
InferenceModule(element, l, net, prev){

  
	factor = 0.1f;
	const char* a = element->Attribute("method");
	string str(a ? a : net->DefaultActivation().c_str());

	if (str == "leaky") mode = LEAKY;
	else if (str == "linear") mode = LINEAR;
	else if (str == "logistic") mode = LOGISTIC;
	else if (str == "relu") mode = RELU;
	else if (str == "lhtan") mode = LHTAN;
	else if (str == "hardtan") mode = HARDTAN;
	else if (str == "tanh") mode = TANH;
	else if (str == "loggy") mode = LOGGY;
	else if (str == "elu") mode = ELU; 
	else   mode = LEAKY;  
	GetPrevModules(element);
	output_width = input_width;
	output_height = input_height;
	output_channels = input_channels;
}

ActivationModule::~ActivationModule() {
}

bool ActivationModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false; 
	return activate_array_ongpu(input, output, input.Elements(), input.DataType(), mode);
}
bool ActivationModule::Backward(CudaTensor & delta) {	
	if (!InferenceModule::Backward(delta)) return false; 
	if (!activate_array_ongpu(output, delta, output.Elements(), output.DataType(), mode)) return false;
	return DistributeDeltas(delta);
}
bool ActivationModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const {
 
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, fp16)) return false;
	string t;
	switch (mode) {
	case CUDNN_ACTIVATION_RELU: 
		xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"ReLU\">" << endl;
		xml << "      <data negative_slope=\"" << factor << "\" />" << endl;
		break;
	case CUDNN_ACTIVATION_SIGMOID:
		xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" >" << endl;
		xml << "      <data type=\"sigmoid\" />" << endl;
		break;
	default:
		cerr << " Error: Activation type other than sigmoid or Relu is not supported yet!\n";
		return false;
	} 
	
	WritePorts(xml);
	xml << "    </layer>" << endl;
	return true;
}
uint32_t ActivationModule::GetFlops() const {
	return 0;
}

UpSampleModule::UpSampleModule(const XMLElement * element, Layer * l, CNNNetwork* network, InferenceModule* prev) :
	InferenceModule(element, l, network, prev) {
 
	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2); 
 
	GetPrevModules(element);
	output_width = input_width * stride_w;
	output_height = input_height * stride_h;
	output_channels = input_channels;
}

UpSampleModule::~UpSampleModule() {
}
bool UpSampleModule::Resize(int w,int h) {
	input_height = h;
	input_width = w;
	output_width = input_width * stride_w;
	output_height = input_height * stride_h; 
	return true;
}
bool UpSampleModule::Forward( ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	bool ret = input.UpSample(output,stride_w,stride_h);
	return ret;
}

bool UpSampleModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	CudaTensor temp(network->DataType(), network->DataFormat());
	temp = delta;
	if(!delta.Init(input.Batch(), input.Channel(), input.Height(), input.Width())) {
		return false;
	}
	if (!temp.DownSample(delta, stride_w, stride_h)) return false; 
	return DistributeDeltas(delta);
}
bool UpSampleModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, fp16)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"Resample\">" << endl;
	//<data />
	xml << "      <data antialias=\"0\" factor=\""<< stride_w <<"\" type=\"caffe.ResampleParameter.NEAREST\" />" << endl;
	WritePorts(xml);
	xml << "    </layer>" << endl;
	return true;
}
uint32_t UpSampleModule::GetFlops() const {
	return 0;
}
DeconvModule::DeconvModule(const XMLElement * element, Layer * l, CNNNetwork* network, InferenceModule* prev) : InferenceModule(element, l, network, prev) {
 
}

DeconvModule::~DeconvModule() {
}

bool DeconvModule::Forward(ForwardContext & context) {
	//TODO: 
	return false;
}

bool DeconvModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	return DistributeDeltas(delta); 
}
