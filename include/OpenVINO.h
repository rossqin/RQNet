#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
using namespace std;
struct OpenVINOIRv7Port {
	int id;
	int batch;
	int channel;
	int height;
	int width;
};
ostream& operator<<(ostream& os, const OpenVINOIRv7Port& port);
struct OpenVINOIRv7ParamsNode {
	string name;
	int indent;
	map<string, string> params; 
	OpenVINOIRv7ParamsNode() { indent = 3; name = "data"; }
	
};
ostream& operator<<(ostream& os, const OpenVINOIRv7ParamsNode& node);
class InferenceModule;
struct OpenVINOIRv7Layer {
	int id;
	string name;
	string precision;
	string ltype;
	const InferenceModule* using_module;
	 
	vector<OpenVINOIRv7ParamsNode> param_nodes;
	vector<OpenVINOIRv7Port> inputs;
	vector<OpenVINOIRv7Port> outputs;
	OpenVINOIRv7Layer(int _id, const string& _name, const char* t, const char* _prec = "FP16");
};
ostream& operator<<(ostream& os, const OpenVINOIRv7Layer& layer);
struct OpenVINOIRv7Edge {
	int from_layer;
	int from_port;
	int to_layer;
	int to_port;
};
ostream& operator<<(ostream& os, const OpenVINOIRv7Edge& edge);