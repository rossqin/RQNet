#include "stdafx.h"

#include "network.h"
#include "inference_module.h"

Layer::Layer(const XMLElement* element,int i, InferenceModule*& prev_module) {
	index = i; 
	name = element->Attribute("id");
	if (name.length() == 0) {
		char buf[20];
		sprintf(buf, "layer%02d", i);
		name = buf;
	}
	last_module = NULL;
	const XMLElement* moduleElement = element->FirstChildElement("module");
	while (moduleElement){
		string mtype = moduleElement->Attribute("type");
		prev_module = InferenceModule::FromXmlElement(moduleElement,this,GetNetwork().GetDataOrder(), prev_module);
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
		char filename[MAX_PATH];
		sprintf(filename, "%s%s.output.bin", DEBUGGING_DIR, name.c_str());
		//modules[i]->output.SaveBatchData(filename, -1);
		//n = i; 
	}
#if 0
	if (n >= 0) {
		char filename[MAX_PATH];
		sprintf(filename, "E:\\AI\\Data\\debugging\\RQNet\\%s.forward.bin", name.c_str());
		ofstream of(filename, ios::binary | ios::trunc);
		FloatTensor4D& tensor = modules[n]->output;
		int bytes = tensor.Elements3D() * sizeof(float);
		char* data = new char[bytes];
		float* src = tensor.GetMem() + tensor.Elements3D();
		cudaMemcpy(data, reinterpret_cast<char*>(src), bytes, cudaMemcpyDeviceToHost);
		if (of.is_open()) {
			of.write(data, bytes);
			of.close();
		}
		delete[]data;
	}
#endif
	return true;
}

bool Layer::Backward(FloatTensor4D & delta) {
	for (int i = modules.size() - 1; i >= 0; i--) {		
		if (!modules[i]->Backward(delta)) return false;

	}
#if 0
	char filename[MAX_PATH];
	sprintf(filename, "E:\\AI\\Data\\debugging\\RQNet\\%s.backward.bin", name.c_str());
	ofstream of(filename, ios::binary | ios::trunc); 
	int bytes = delta.Elements3D() * sizeof(float);
	char* data = new char[bytes];
	cudaMemcpy(data, delta.GetMem(), bytes, cudaMemcpyDeviceToHost);
	if (of.is_open()) {
		of.write(data, bytes);
		of.close();
	}
	delete[]data;
#endif
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
