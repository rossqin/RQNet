#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "cuda_tensor.h"
#include "yolo.h"
#include "inference_module.h"
#include "config.h"
const float one = 1.0f, zero = 0.0f;
InferenceModule* InferenceModule::last_parsing = nullptr;
InferenceModule::InferenceModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev) 
	: input(net->DataType(), net->DataFormat()), output(net->DataType(), net->DataFormat()), 
	shortcut_delta(net->DataType(), net->DataFormat()) {

	network = net;
	ir_output_port = 1; 
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
	if (!id) id = "conv1"; 
	name = base + id;
	//cout << " INFO: Initializing " << name << " ...\n";
	concat_prevs = true;
	ir_type = nullptr; 
}
 
void InferenceModule::ParsePrevModules(const XMLElement* element ) {
	
	if (!last_parsing) {
		input_width = network->input_width;
		input_height = network->input_height;
		input_channels = network->input_channels; 
		return;
	} 
	const char* before = element->Attribute("before");
	if (!before) {
		input_width = last_parsing->output_width;
		input_height = last_parsing->output_height;
		input_channels = last_parsing->output_channels;
		prevs.push_back({ last_parsing, -1 });
		return;
	}
	
	vector<string> befores;
	split_string(befores, string(before));
	input_channels = 0;
	input_width = 0;
	input_height = 0;
	for (string& s : befores) {
		size_t pos = s.find('[', 1);		
		string sub_channels;
		Layer* prev_layer = nullptr;
		InferenceModule* module = nullptr;
		string mname;
		if (pos != string::npos) {
			++pos;
			size_t pos2 = s.find(']', pos );
			if (pos2 == string::npos) pos2 = s.length();
			sub_channels = s.substr(pos, pos2 - pos);
			s.erase(pos-1); 
		}
		pos = s.find('.', 1);
		if (pos != string::npos) {
			string lname = s.substr(0, pos);
			mname = s.substr(pos + 1);
			if (_strcmpi("prev_layer", lname.c_str()) == 0) {
				prev_layer = *(network->layers.rbegin());
			}
			else {
				for (Layer* lf : network->layers) {
					if (lname == lf->GetName()) {
						prev_layer = lf;
						break;
					}
				}
			}
		}
		else {
			prev_layer = layer;
			mname = s;
		}
		if (!prev_layer) {
			stringstream ss;
			ss << "No layer found for `" << s << "` !";
			throw std::exception(ss.str().c_str());
		}
		if (_strcmpi("last", mname.c_str()) == 0) {
			module = *(prev_layer->modules.rbegin());
		}
		else {
			mname = prev_layer->GetName() + "." + mname;
			for (auto& m : prev_layer->modules) {
				if (mname == m->name) {
					module = m;
					break;
				}
			}
		}
		if (!module) {
			stringstream ss;
			ss << "No module found with the name of `" << s << "` !";
			throw std::exception(ss.str().c_str());
		}
		
		if (input_width < module->output_width) input_width = module->output_width;
		if (input_height < module->output_height) input_height = module->output_height;
		if (sub_channels.empty()) {
			prevs.push_back({ module, -1 });
			
			if (concat_prevs)
				input_channels += module->output_channels;
			else if (input_channels < module->output_channels)
					input_channels = module->output_channels;
			 
		}
		else {
			vector<int> group_ids;
			parse_integers(group_ids, sub_channels);
			for (int& id : group_ids) { 
				prevs.push_back({ module, id });
				if (concat_prevs)
					input_channels += module->output_channels;
				else if (input_channels < module->output_channels) input_channels = module->output_channels;
				 
			}
		}
	}
}
 
InferenceModule* InferenceModule::FromXmlElement(const XMLElement* element,  Layer* layer, CNNNetwork* network, InferenceModule* prev) {
	const char *t = element->Attribute("type");
	string mtype(t ? t : "conv");
	InferenceModule* module = nullptr;
	if (!t) {
		cout << " INFO: *** not `type` definition, set to `conv`.\n";
	}
	if (mtype == "conv" || mtype =="dwconv" || mtype == "convolutional" ) {
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
	else if (mtype == "yolo-detection" || mtype == "yolo") {
		module = New YoloModule(element, layer, network, prev);
	}
	else if (mtype == "eltwise") {
		module = New EltwiseModule(element, layer, network, prev);
	}
	else if (mtype == "concat") {  // special 
		module = New ConcatModule(element, layer, network, prev); 
	}
	else if (mtype == "split") {  // special 
		module = New SplitModule(element, layer, network, prev);
	}
	else if(mtype =="shuffle"){
		module = New ShuffleModule(element, layer, network, prev);
	}
	//TODO: Add your New Moule here.

	if (module)
		network->module_pool[module->name] = module; 
	else
		throw t;
	last_parsing = module;
	return module;
}
 
bool InferenceModule::Forward(ForwardContext & context) {
	int n = (int)prevs.size();  
	int w = 0;
	int h = 0;
	context.input = nullptr; 
	if (n == 1) { 
		InferenceModule* module = prevs[0].module;
		w = module->output_width;
		h = module->output_height;
		context.input = &(module->GetOutput(prevs[0].group_id));
	}
	else if (n > 1) {
		int channels = 0;
		for (auto& p : prevs) {
			if (w < p.module->output_width) w = p.module->output_width;
			if (h < p.module->output_height) h = p.module->output_height;
			channels += p.module->output_channels;
		}
		if (channels != input_channels) {
			throw exception("input channel error!");
		}
		if (!input.Init({ network->mini_batch, input_channels, h, w })) {
			cerr << name << ".forward failed as input intialization failed!\n";
			return false;
		}
		size_t offset = 0;
		for(int i = 0 ; i < prevs.size() ; i++){
			CudaTensor& o = prevs[i].module->GetOutput(prevs[i].group_id);
			size_t batch_bytes = o.Elements3D() * o.ElementBytes();
			for (int b = 0; b < network->mini_batch; b++) {
				char* dest = input.BatchData(b) + offset;
				if (cudaSuccess != cudaMemcpy(dest, o.BatchData(b), batch_bytes, cudaMemcpyDeviceToDevice)) {
					cerr << "Error: " << name << ".forward failed due to input filling!\n";
					return false;
				}
			}
			offset += batch_bytes;
		}
		context.input = &input;
	}
	else { 	// n == 0, this is the first module
		w = network->input_width;
		h = network->input_height;
		int expected_elements = network->mini_batch * network->input_channels * w * h;
		if (input_width != w || input_height != h || input.Elements() != expected_elements) {
			if (!input.Init({ network->mini_batch, input_channels, w, h })) {
				return false;
			}
		}
		if (!input.Push(network->input)) {
			return false;
		}
		context.input = &input;
	}
	if (input_width != w || input_height != h) {
		if (!Resize(w, h)) {
			return false;
		}
	}
	if (dynamic_cast<SplitModule*>(this)) return true;
	int expected_output_elemens = network->mini_batch * output_channels * output_height * output_width;
	if (output.Elements() != expected_output_elemens &&
		!output.Init({ network->mini_batch, output_channels, output_height, output_width })) {
		return false;
	}
 
	return true;
}
 
bool InferenceModule::Backward(CudaTensor & delta) {
	if (shortcut_delta.Elements() != 0) {
		if (delta.Like(shortcut_delta)) {
			if (!delta.Add(shortcut_delta)) return false;
		}
		else 
			delta = shortcut_delta;
		return shortcut_delta.Release();
	}

	return true;
}
bool InferenceModule::ShortcutDelta(const CudaTensor& d, int group_id) {
	TensorDims dim = shortcut_delta.Dims();
	
	if (dim.n != network->mini_batch || dim.c != output_channels || dim.h != output_height || dim.w != output_width) {
		if (!shortcut_delta.Init({ network->mini_batch , output_channels , output_height , output_width }))
			return false;
	}
	return shortcut_delta.Add(d);
}

bool InferenceModule::DistributeDeltas(CudaTensor & delta) {
	if (prevs.size() == 0) return true;
	if (!concat_prevs) return false; 
	size_t fm_size = input_height * input_width * delta.ElementBytes();
	size_t offset = 0, bytes_to_copy = 0;
	switch (prevs.size()) {
	case 1:
		if (prevs[0].group_id < 0 && prevs[0].module == logical_prev) {
			// input_channels == output_channels ;
			return true;
		}
		if (!prevs[0].module->ShortcutDelta(delta, prevs[0].group_id)) {
			cerr << "Error: " << name << ".backward.DistributeDeltas failed !\n";
			return false;// input_channels != output_channels ;
		}
		return delta.Release();
	case 0:

		return true;
	default:
		for (int n = 0; n < prevs.size(); n++) {
			InferenceModule* m = prevs[n].module;
			CudaTensor temp(network->DataType(),network->DataFormat());
			if (!temp.Init({ network->mini_batch, m->output_channels, m->output_height, output_width})) {
				cerr << "Error: " << name << ".backward.DistributeDeltas failed(init) !\n";
				return false;// input_channels != output_channels ;
			}
			bytes_to_copy = m->output_channels * fm_size;
			for (int b = 0; b < network->mini_batch; b++) {
				char* src = delta.BatchData(b) + offset ;
				char* dest = temp.BatchData(b);				
				if (cudaSuccess != cudaMemcpy(dest, src, bytes_to_copy, cudaMemcpyDeviceToDevice)) {
					cerr << "Error: " << name << ".backward.DistributeDeltas failed (memcpy) !\n";
					return false;
				}
			}
			if (!m->ShortcutDelta(temp, prevs[n].group_id)) {
				cerr << "Error: " << name << ".backward.DistributeDeltas failed !\n";
				return false;
			}
			offset += bytes_to_copy;
		} 
		return delta.Release(); 
	} 

	return true; 
}
bool InferenceModule::RenderOpenVINOIR(vector<OpenVINOIRv7Layer>& layers, vector<OpenVINOIRv7Edge>& edges,
	ofstream& bin, size_t& bin_offset, bool fp16) {
	if (!ir_type) {
		return false;
	}
	OpenVINOIRv7Layer this_layer(layers.size(), name, ir_type, fp16 ? "FP16" : "FP32" );
	this_layer.using_module = this; 
	string key, attr;
	for (auto& p : ir_params) {
		size_t pos = p.first.find('.', 0);
		if (pos == string::npos) {
			key = "data";
			attr = p.first;
		}
		else {
			key = p.first.substr(0, pos);
			attr = p.first.substr(pos + 1);
		}
		bool new_node = true;
		for (auto& lp : this_layer.param_nodes) {
			if (lp.name == key) {
				lp.params[attr] = p.second;
				new_node = false;
				break;
			}
		}
		if (new_node) {
			OpenVINOIRv7ParamsNode node;
			node.name = key;
			node.params[attr] = p.second;
			this_layer.param_nodes.push_back(node);
		}
	}
	if (prevs.size() > 0) {
		if (prevs.size() == 1) {
			this_layer.inputs.push_back({ 0,1, input_channels, input_height, input_width }); 
			int group_id = -1;
			InferenceModule* m = GetPrev(0,group_id);			  
			for (auto& l : layers) {
				if (l.using_module == m) {
					edges.push_back({l.id, (group_id < 0) ? m->ir_output_port : (m->ir_output_port + group_id), this_layer.id, 0});
					break;
				}
			}
		 }
		else {
			bool add_virtual_concat = (dynamic_cast<ConcatModule*>(this) == nullptr) && (dynamic_cast<EltwiseModule*>(this) == nullptr);
	 
			OpenVINOIRv7Layer* pl = &this_layer;
			if (add_virtual_concat) {
				pl = New OpenVINOIRv7Layer(this_layer.id++, name + ".concat", "Concat", fp16 ? "FP16" : "FP32");
			}
			int pid = 0;
			for (int i = 0; i < prevs.size(); i++) {
				int group_id = -1;
				InferenceModule* m = GetPrev(i, group_id); 
				pl->inputs.push_back({ pid, 1, m->output_channels , m->output_height, m->output_width });
				for (auto& l : layers) {
					if (l.using_module == m) {
						edges.push_back({ l.id, (group_id < 0) ? m->ir_output_port :(m->ir_output_port + group_id) , pl->id, pid });
						break;
					}
				}
				pid++;
			}			
			if (add_virtual_concat) {
				pl->outputs.push_back({ pid,1,input_channels,input_height, input_width });
				layers.push_back(*pl);
				this_layer.inputs.push_back({ 0,1, input_channels, input_height, input_width });
				edges.push_back({pl->id, pl->outputs[0].id, this_layer.id,0});
				delete pl;
			}

			 
		}
	}
	else {
		// the first layer
		this_layer.inputs.push_back({ 0,1, input_channels, input_height, input_width });
		edges.push_back({ 0, 0 , this_layer.id, 0 });
	}
	int start_id = this_layer.inputs.size();
	ir_output_port = start_id;
	int output_count = OutputCount(); 
	for (int g = 0; g < output_count; g++) {
		this_layer.outputs.push_back({ start_id++ ,1, output_channels, output_height, output_width });
	}
	 
	
	layers.push_back(this_layer);
	return true;
}
InferenceModule* InferenceModule::GetPrev(int n, int& group_id, bool ignore_bn) const {
	if (prevs.size() == 0 || n < 0 || n >= prevs.size()) return nullptr;
	InferenceModule* ret = prevs[n].module;
	group_id = prevs[n].group_id;
	if (!ignore_bn) return ret;
	while (ret) {
		if (!dynamic_cast<BatchNormModule*>(ret)) return ret;
		group_id = ret->prevs[0].group_id;
		ret = ret->prevs[0].module;
		
	}
	return ret;
}
bool InferenceModule::CheckRedundantChannels(float c_threshold, float w_threshold) {

	valid_channels.assign(input_channels, true);
	if (prevs.size() == 1) {
		if (prevs[0].group_id == -1)
			valid_channels = prevs[0].module->valid_channels;
		else {
			cout << " Hint: partial connection to " << name << ", pruning may be failed.\n";
		}
	}
	else {
		int index = 0;
		valid_channels.assign(input_channels, true);
		for (int i = 0; i < prevs.size(); i++) {
			auto& p = prevs[i];
			if (p.group_id == -1) {
				for (int j = 0; j < p.module->output_channels; j++) {
					valid_channels[j + index] = p.module->valid_channels[j];
				}
				index += p.module->output_channels;
			}
			else {
				cout << " Hint: partial connection to " << name << ", pruning may be failed.\n";
				index += p.module->GetOutputChannels();
			}
		}
	} 
	return true;
}