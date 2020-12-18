#include "stdafx.h"

#include "network.h"
#include "inference_module.h"

Layer::Layer(const XMLElement* element, CNNNetwork* net, InferenceModule*& prev_module) {
	index = net->layers.size(); 
	const char* s = element->Attribute("id");
	if (!s || !(*s)) {
		char buf[20];
		sprintf(buf, "layer%02d", index);
		name = buf;
	}
	else
		name = s;
	s = element->Attribute("desc");
	if (s && *s) desc = s; 
	network = net;
	const XMLElement* moduleElement = element->FirstChildElement("module");
	while (moduleElement){		
		prev_module = InferenceModule::FromXmlElement(moduleElement, this, net,  prev_module);
		modules.push_back(prev_module);
		moduleElement = moduleElement->NextSiblingElement();
	} 
} 
bool Layer::Forward(ForwardContext & context) { 
	for(size_t i = 0 ; i < modules.size(); i++){
		InferenceModule* module = modules[i];
		if (!module->Forward(context)) {
			cerr << "Forward failed at " << modules[i]->Name() << endl;
			return false;
		}		
	}
	return true;
}

bool Layer::Backward(CudaTensor & delta) {
	for (int i = (int)modules.size() - 1; i >= 0; i--) {		
		InferenceModule* module = modules[i];
		if (!module->Backward(delta)) {
			return false;
		}
		//char filename[MAX_PATH];
		//sprintf(filename, "%s0.2.0\\%s.delta.bin", DEBUGGING_DIR, module->name.c_str());
		//delta.Save(filename,1);
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
	return true;
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