#include "OpenVINO.h"
#include "inference_module.h"
 
ostream& operator<<(ostream& os, const OpenVINOIRv7ParamsNode& node) {
	string s;
	if (node.indent > 1) {		
		s.assign(node.indent, '\t');
		os << s;
	}
	os << "<" << node.name;
	for (auto& p : node.params) {
		os << " " << p.first << "=\"" << p.second << "\"";
	} 
	os << "/>\n"; 

	return os;
}

OpenVINOIRv7Layer::OpenVINOIRv7Layer(int _id, const string& _name,const char* t, const char* _prec) {
	id = _id;
	name = _name;
	if(t) ltype = t;
	if (_prec) precision = _prec;
	using_module = nullptr;
}
//fore to be NCHW
ostream& operator<<(ostream& os, const OpenVINOIRv7Port& port) {
	os << "\t\t\t\t<port id=\"" << port.id << "\" >\n";
	os << "\t\t\t\t\t<dim>" << port.batch << "</dim>\n";
	os << "\t\t\t\t\t<dim>" << port.channel << "</dim>\n";
	os << "\t\t\t\t\t<dim>" << port.height << "</dim>\n";
	os << "\t\t\t\t\t<dim>" << port.width << "</dim>\n";
	os << "\t\t\t\t</port>\n";
	return os;
}
ostream& operator<<(ostream& os, const OpenVINOIRv7Layer& layer) {
	os << "\t\t<layer id=\"" << layer.id << "\" name=\"" << layer.name << "\" precision=\""
		<< layer.precision << "\" type=\"" << layer.ltype << "\" >\n";
	if (layer.inputs.size() > 0) {
		os << "\t\t\t<input>\n";
		for (auto& p : layer.inputs) {
			os << p;
		}
		os << "\t\t\t</input>\n";
	} 
	os << "\t\t\t<output>\n";
	for (auto& p : layer.outputs) {
		os << p;
	}
	os << "\t\t\t</output>\n";
	for (auto& p : layer.params) {
		os << p;
	}
	os << "\t\t</layer>\n";
	return os;
}
ostream& operator<<(ostream& os, const OpenVINOIRv7Edge& edge) {
	os << "\t\t<edge  from-layer=\""<< edge.from_layer << "\" from-port=\""<< edge.from_port 
		<<"\" to-layer=\"" << edge.to_layer << "\" to-port=\"" << edge.to_port <<  "\"/>\n";
	return os;
}