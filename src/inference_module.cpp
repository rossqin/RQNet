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
	if (nullptr == id) id = "convolution";
	name = base + id;
	cout << " INFO: Initializing " << name << " ...\n";
	cached_input = nullptr;
	cached_output = nullptr;
	if(prev) prev->logical_next = this;
	logical_next = nullptr;
}
InferenceModule::~InferenceModule() { 
	if (cached_input) delete[]cached_input;
	if (cached_output) delete[]cached_output;
}
void InferenceModule::GetPrevModules(const XMLElement* element, bool add_channels ) {
	string base = layer->GetName() + "."; 
	const char *p = element->Attribute("before");	 
	string prev_ids_s(p ? p : "none");
	input_channels = 0;
		 
	if (prev_ids_s.length() > 0 && prev_ids_s != "none") {
		vector<string> prev_ids;
		ModulePool::iterator  it;
		split_string(prev_ids, prev_ids_s);		
		for (auto id : prev_ids) {
			if (id.find('.') == string::npos)
				it = network->module_pool.find(base + id);
			else
				it = network->module_pool.find(id);
			if (it != network->module_pool.end()) {
				InferenceModule* module = it->second ;
				if(add_channels)
					input_channels += module->output_channels;
				else {
					if (input_channels < module->output_channels)
						input_channels = module->output_channels;
				}
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
	InferenceModule* module = nullptr;
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
	else if (mtype == "shortcut") {
		module = New ShortcutModule(element, layer, network, prev);
	}
	//TODO: Add your New Moule here.

	if (module)
		network->module_pool.insert(pair<string, InferenceModule*>(module->name, module));
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
	return shortcut_delta.Init(output.Batch(), output_channels, output_height, output_width);
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
		InferenceModule* module = prevs[0];
		BatchNormModule* bn_module = dynamic_cast<BatchNormModule*>(module);
		if (!bn_module || !bn_module->IsFused()) {
			context.input = &(module->output);
		}		
		w = context.input->Width();
		h = context.input->Height();
		
	}
	else if (n > 1) { 
		vector<const CudaTensor*> srcs;
		for (int i = 0; i < n; i++) {
			if (prevs[i]->output_height > h)
				h = prevs[i]->output_height;
			if (prevs[i]->output_width > w)
				w = prevs[i]->output_width;
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
		context.input = &input;
	}
	else { 
		
		w = network->input_width;
		h = network->input_height;
		int expected_elements = network->mini_batch * network->input_channels * w * h;
		if (input_width != w || input_height != h || input.Elements() != expected_elements) {
			if (!input.Init(network->mini_batch, input_channels, w, h)) return false; 
		}
		if (!input.Push(network->input)) return false;
		context.input = &input;
	}
	if (input_width != w || input_height != h) {
		if (!Resize(w, h)) return false; 
			
	}
	int expected_output_elemens = network->mini_batch * output_channels * output_height * output_width;
	if (output.Elements() != expected_output_elemens &&
		!output.Init(network->mini_batch, output_channels, output_height, output_width)) return false;
 
	return true;
}
 
bool InferenceModule::Backward(CudaTensor & delta) {
	if (shortcut_delta.Elements() != 0) {
		if (delta.SameShape(shortcut_delta)) {
			if (!delta.Add(shortcut_delta)) return false;
		}
		else 
			delta = shortcut_delta;
		return shortcut_delta.Release();
	}

	return true;
}
bool InferenceModule::DistributeDeltas(CudaTensor & delta) {
	int n = (int)prevs.size();
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
	return true;//delta.Release(); 
}

bool InferenceModule::OutputIRModel(ofstream & xml, ofstream & bin, stringstream & edges, size_t & bin_offset, int &l_index) const {
	int n = (int)prevs.size();
	if (0 == n) {
		edges << "    <edge from-layer=\"0\" from-port=\"0\" to-layer=\"" << index 
			<< "\" to-port=\"0\"/>" << endl;
	} 
	if (n > 1) {
		index = l_index + 1;
	}
	else {
		index = l_index;
	}
	for (int i = 0; i < n; i++ ) {
		InferenceModule* module = prevs[i];
		int from_port = module->output_port;
		int from_index = module->index;
		BatchNormModule* bn_module = dynamic_cast<BatchNormModule*>(module);
		if (bn_module && bn_module->IsFused()) {
			ConvolutionalModule* con_module = dynamic_cast<ConvolutionalModule*>(bn_module->prevs[0]);
			if (!con_module) {
				cerr << " Error: Fused Batchnorm Module does not follow Convolutional Module!\n";
				return false;
			}
			module = con_module;
			from_port = module->output_port;
			from_index = module->index;
		}
		edges << "    <edge from-layer=\"" << from_index << "\" from-port=\"" << from_port <<
			"\" to-layer=\"" << l_index << "\" to-port=\"" << i << "\"/>" << endl;
	}
	
	if (n < 2) {
		l_index = index + 1;
		return true;
	}
	string concat_name = name + ".concat";
	xml << "    <layer id=\"" << l_index << "\" name=\"" << concat_name << "\" precision=\"" << Precision() << "\" type=\"Concat\">" << endl;
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
	xml << "      <output>" << endl;
	xml << "        <port id = \"" << n << "\">" << endl;
	xml << "          <dim>1</dim>" << endl;
	xml << "          <dim>" << input_channels << "</dim>" << endl;
	xml << "          <dim>" << input_height << "</dim>" << endl;
	xml << "          <dim>" << input_width << "</dim>" << endl;
	xml << "        </port>" << endl;
	xml << "      </output>" << endl;
	xml << "    </layer>" << endl;
	edges << "    <edge from-layer=\"" << l_index  << "\" from-port=\"" << n <<
		"\" to-layer=\"" << index << "\" to-port=\"0\"/>" << endl;
	l_index = index + 1;
	return true;
}

bool InferenceModule::CacheOutput() {
// 	if (output.Bytes() > 0 && output.Data() != nullptr && !cached_output) {
// 		return output.Cache(cached_output);
// 	}
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
	return activate_array_ongpu(context.input->Data(), output, output.Elements(), output.DataType(), mode);
}
bool ActivationModule::Backward(CudaTensor & delta) {	
	if (!InferenceModule::Backward(delta)) return false;
	//delta.Save(DEBUGGING_DIR "activation.before02.bin", 1);
	if (!gradient_array_ongpu(output, delta, output.Elements(), output.DataType(), mode)) return false;
	//delta.Save(DEBUGGING_DIR "activation.after02.bin", 1);
	return DistributeDeltas(delta);
}
bool ActivationModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const {
 
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, l_index)) return false;
	string t;
	switch (mode) {
	case  LEAKY:
	case RELU:
		xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"ReLU\">" << endl;
		xml << "      <data negative_slope=\"" << factor << "\" />" << endl;
		break;
	case LOGISTIC:
		xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" >" << endl;
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
	return context.input->UpSample(output,stride_w,stride_h); 
}

bool UpSampleModule::Backward(CudaTensor & delta) {
	if (!InferenceModule::Backward(delta)) return false;
	CudaTensor temp(network->DataType(), network->DataFormat());
	temp = delta;
	if(!delta.Init(network->MiniBatch(),input_channels, input_height, input_width)) {
		return false;
	}
	if (!temp.DownSample(delta, stride_w, stride_h)) return false; 
	return DistributeDeltas(delta);
}
bool UpSampleModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset,int& l_index) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, l_index)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"Resample\">" << endl;
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

bool ShortcutModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height;
	return true;
}

ShortcutModule::ShortcutModule(const XMLElement * element, Layer * l, CNNNetwork * net, InferenceModule * prev) :
InferenceModule(element,l,net,prev){
	GetPrevModules(element,false);
	output_height = input_height;
	output_width = input_width;
	output_channels = input_channels;
}

ShortcutModule::~ShortcutModule()
{
}

bool ShortcutModule::Forward(ForwardContext & context) {
	int n = (int)prevs.size(); 
	if (n < 2) return false; 
	int expected_elements = network->MiniBatch() * output_channels * output_width * output_height;
	output = prevs[0]->output;
	if (output.Elements() != expected_elements) return false;
	for (int i = 1; i < n; i++) {
		InferenceModule* prev = prevs[i];
		if (prev->output.Elements() != expected_elements) {
			return false;
		}
		if (!output.Add(prev->output)) return false;
 	}
	return true;
}

bool ShortcutModule::Backward(CudaTensor & delta) {
	if(!InferenceModule::Backward(delta)) return false;
	int n = (int)prevs.size();
	if (n < 2) return false; 
	for (int i = 0; i < n; i++) {
		InferenceModule* module = prevs[i]; 
		if (module == logical_prev) {
			continue;
		}
		if (module->shortcut_delta.Elements() == 0)
			module->shortcut_delta = delta;
		else if (module->shortcut_delta.SameShape(delta))
			return module->shortcut_delta.Add(delta);
		else
			return false;
	}
	return true;
}

bool ShortcutModule::OutputIRModel(ofstream & xml, ofstream & bin, stringstream & edges, 
	size_t & bin_offset, int & l_index) const {	
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"Eltwise\">" << endl;
	xml << "      <data operation=\"sum\" />"<<endl;
	
	int i = 0; 
	xml << "      <input>" << endl;
	
	while (i < (int)prevs.size()) {
		InferenceModule* prev = prevs[i]; 
		xml << "        <port id=\"" << i << "\">" << endl;
		xml << "          <dim>1</dim>" << endl;
		xml << "          <dim>" << prev->output_channels << "</dim>" << endl;
		xml << "          <dim>" << prev->output_height << "</dim>" << endl;
		xml << "          <dim>" << prev->output_width << "</dim>" << endl;
		xml << "        </port>" << endl;
		edges << "    <edge from-layer=\"" << prev->index << "\" from-port=\"" << prev->output_port <<
			"\" to-layer=\"" << index << "\" to-port=\"" << i << "\"/>" << endl;
		i++;
	}
	xml << "      </input>" << endl;
	xml << "      <output>" << endl;
 
	xml << "        <port id=\"" << output_port << "\">" << endl;
	xml << "          <dim>1</dim>" << endl;
	xml << "          <dim>" << output_channels << "</dim>" << endl;
	xml << "          <dim>" << output_height << "</dim>" << endl;
	xml << "          <dim>" << output_width << "</dim>" << endl;
	xml << "        </port>" << endl;
	xml << "      </output>" << endl;
	 
	xml << "    </layer>" << endl;
	return true; 
}
