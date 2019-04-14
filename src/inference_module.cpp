#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "tensor.h"
#include "yolo.h"
#include "inference_module.h"
#include "config.h"
const float one = 1.0f, zero = 0.0f;
InferenceModule::InferenceModule(const XMLElement* element, Layer* l, InferenceModule* prev) {
	if (prev)
		index = prev->index + 1;
	else
		index = 1;
	output_port = 3;
	layer = l;
	y_desc = NULL;
	x_desc = NULL;
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
		input_channels = GetNetwork().GetInputChannels();
		input_width = GetNetwork().GetInputWidth();
		input_height = GetNetwork().GetInputHeight();
	}
	
	if (prevs.size() > 1) {
		index ++; // save for later IR transform
	}
}
 
InferenceModule* InferenceModule::FromXmlElement(const XMLElement* element,  Layer* layer, InferenceModule* prev) {
	const char *t = element->Attribute("type");
	string mtype(t ? t : "");
	InferenceModule* module = NULL;
	if (mtype == "conv" || mtype == "convolutional") {
		module = New ConvolutionalModule(element, layer, prev);
	}
	else if (mtype == "batch-norm" ) {
		module = New BatchNormModule(element, layer, prev);
	}
	else if(mtype == "activation") {
		module = New ActivationModule(element, layer, prev);
	}
	else if (mtype == "max-pool" || mtype == "avg-pool") {
		module = New PoolingModule(element, layer, prev);
	}
	else if (mtype == "upsample") {
		module = New UpSampleModule(element, layer, prev);
	}
	else if (mtype == "yolo-detection") {
		module = New YoloModule(element, layer, prev);
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
	int batch = GetAppConfig().GetMiniBatch();
	//first time, w == 0, h == 0;

	int w = 0;
	int h = 0;

	if (n == 1) {
		input = prevs[0]->output;
		w = input.GetWidth();
		h = input.GetHeight();
	}
	else if (n > 1) {
		input = 0.0f;		
		TensorOrder to = GetNetwork().GetDataOrder();
		for (int i = 0; i < n; i++) {
			if (prevs[i]->output.GetHeight() > h)
				h = prevs[i]->output.GetHeight();
			if (prevs[i]->output.GetWidth() > w)
				w = prevs[i]->output.GetWidth();
		} 
		if (w != input_width || h != input_height) {
			if (!input.Init(batch, input_channels, w, h, to)) return false;
		}
		else {
			input = 0.0f;
		}
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
	else {
		input = context.input;
		w = input.GetWidth();
		h = input.GetHeight();
	}
	if (input_width != w || input_height != h) {
		input_width = w;
		input_height = h;
		if (!InitDescriptors()) return false; 
		if (!output.Init(batch, output_channels, output_width, output_height, input.GetOrder())) return false;
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
 
bool ActivationModule::InitDescriptors() { 
	output_width = input_width; 
	output_height = input_height; 
	return true; 
}

ActivationModule::ActivationModule(const XMLElement* element, Layer* l, InferenceModule* prev):
InferenceModule(element,l,prev){
	factor = 0.1f;
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
	InitDescriptors();
	output_channels = input_channels;
}

bool ActivationModule::Forward(ForwardContext & context) {
	if (!InferenceModule::Forward(context)) return false;
	output = input; 
	bool ret = activate_array_ongpu(output.GetMem(),output.MemElements(),atype); 
	return ret;
}
 
bool ActivationModule::Backward(FloatTensor4D & delta) {
	
	if (!InferenceModule::Backward(delta)) return false;
	if (!output.SameDememsion(delta)) return false;
	if (!gradient_array_ongpu(output.GetMem(),delta.GetMem(),output.MemElements(), atype))
		return false ;
	return DistributeDeltas(delta);
}
bool ActivationModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, bool fp16) const {
 
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, fp16)) return false;
	string t;
	switch (atype) {
	case LEAKY:
	case RELU:
		t = "ReLU";
		break;
	default:
		return false;
	} 
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << (fp16 ? "FP16" : "FP32") << "\" type=\"" << t << "\">" << endl;
	xml << "      <data negative_slope=\"" << factor << "\" />" << endl;
	WritePorts(xml);
	xml << "    </layer>" << endl;
	return true;
}
uint32_t ActivationModule::GetFlops() const {
	return 0;
}

UpSampleModule::UpSampleModule(const XMLElement * element, Layer * l, InferenceModule* prev) :
	InferenceModule(element, l, prev) {
 
	stride_w = element->IntAttribute("stride-w", 2);
	stride_h = element->IntAttribute("stride-h", 2); 
 
	GetPrevModules(element);
	InitDescriptors();
	output_channels = input_channels;
}

UpSampleModule::~UpSampleModule() {
}
bool UpSampleModule::InitDescriptors() {
 
	output_width = input_width * stride_w;
	output_height = input_height * stride_h; 
	return true;
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
DeconvModule::DeconvModule(const XMLElement * element, Layer * l, InferenceModule* prev) : InferenceModule(element, l, prev) {
 
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
