#include "stdafx.h"

#include "network.h"
#include "inference_module.h"

Layer::Layer(const XMLElement* element,int i, CNNNetwork* net, InferenceModule*& prev_module) {
	index = i; 
	name = element->Attribute("id");
	if (name.length() == 0) {
		char buf[20];
		sprintf(buf, "layer%02d", i);
		name = buf;
	}
	last_module = NULL;
	network = net;
	const XMLElement* moduleElement = element->FirstChildElement("module");
	while (moduleElement){
		string mtype = moduleElement->Attribute("type");
		prev_module = InferenceModule::FromXmlElement(moduleElement,this, net,  prev_module);
		modules.push_back(prev_module);
		moduleElement = moduleElement->NextSiblingElement();
	}
	last_module = prev_module;
}
bool Layer::Forward(ForwardContext & context) {
	//int n = -1;
	for(size_t i = 0 ; i < modules.size(); i++){
		if (!modules[i]->Forward(context)) {
			cerr << "Forward failed at " << modules[i]->name << endl;
			return false;
		}
		// char filename[MAX_PATH];
		// sprintf(filename, "%s%s.output.bin", DEBUGGING_DIR, name.c_str());
		//modules[i]->output.SaveBatchData(filename, -1);
		//n = i; 
	}
	return true;
}

bool Layer::Backward(CudaTensor & delta) {
	for (int i = (int)modules.size() - 1; i >= 0; i--) {		
		if (!modules[i]->Backward(delta)) return false;

	}
	return true;
}

bool Layer::FuseBatchNormModule() {
	for(int i = 0 ; i < (int)modules.size() ; i++){
		BatchNormModule* bm = dynamic_cast<BatchNormModule*>(modules[i]);
		if (bm) {
			if (!bm->Fuse()) return false;
		} 
	}
	return false;
}

bool Layer::Update(float lr) {
	for (auto m : modules) {
		if(!m->UpdateParams(lr)) return false;
	}
	return true;
}

void Layer::Print() const
{
}

bool Layer::OutputIRModel(ofstream & xml, ofstream & bin, stringstream & edges, size_t & bin_offset, bool fp16) const{
	for (size_t i = 0; i < modules.size(); i++) {
		if (!modules[i]->OutputIRModel(xml, bin, edges, bin_offset, fp16)) 
			return false;
	}
	return true;
}
