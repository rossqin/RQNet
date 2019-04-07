#include "stdafx.h"

#include "network.h"
#include "inference_module.h"

Layer::Layer(const XMLElement* element,int i ) {
	index = i; 
	name = element->Attribute("id");
	if (name.length() == 0) {
		char buf[20];
		sprintf(buf, "layer%02d", i);
		name = buf;
	}
	const XMLElement* moduleElement = element->FirstChildElement("module");
	while (moduleElement){
		string mtype = moduleElement->Attribute("type");
		InferenceModule* module = InferenceModule::FromXmlElement(moduleElement,this,GetNetwork().GetDataOrder());
		modules.push_back(module);
		 
		moduleElement = moduleElement->NextSiblingElement();
	}
}

bool Layer::Forward(ForwardContext & context) {
	 
	for(size_t i = 0 ; i < modules.size(); i++){
		if (!modules[i]->Forward(context)) return false;
	}
	return true;
}

bool Layer::Backward(FloatTensor4D & delta) {
	for (int i = modules.size() - 1; i >= 0; i--) {		
		if (!modules[i]->Backward(delta)) return false;
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
