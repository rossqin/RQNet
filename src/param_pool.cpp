#include "stdafx.h"
#include "tensor.h"
#include "param_pool.h"
#include "config.h"
static ParamPool thePool;

ParamPool& GetParamPool() {
	return thePool;
}

ParamPool::~ParamPool() {
	if (release_mem) {
		for (auto e : params) {
			delete e.second;
		}
	}
}

void ParamPool::Put(string key, FloatTensor4D* tensor) {
	uninit_params.insert(pair<string, FloatTensor4D*>(key, tensor)); 
}

bool ParamPool::Load(const char * filename) {
	ifstream f(filename,ios::binary);
	if (!f.is_open()) return false;
	param_file_header header;
	f.read(reinterpret_cast<char*>(&header), sizeof(param_file_header));
	tensor_order = header.tensor_order;
	iteration = header.seen / GetAppConfig().GetBatch();
	char* str_buf = New char[header.strings_bytes + 1];
	f.read(str_buf, header.strings_bytes);
	str_buf[header.strings_bytes] = 0;
	tensor_data_header data_header;
	string name; 
	while (!f.eof() && uninit_params.size() > 0 ) {
		f.read(reinterpret_cast<char*>(&data_header), sizeof(tensor_data_header));
		if (f.gcount() < sizeof(tensor_data_header)) {
			cerr << "Warning: " << uninit_params.size() << " params are not loaded from weights file!\n";
			break;
		}
		//TODO: check datatype
		name.assign(str_buf + data_header.name_index + 1, static_cast<uint8_t>(str_buf[data_header.name_index]));
		auto it = uninit_params.find(name); 
		
		if (it != uninit_params.end()) {
			FloatTensor4D* tensor = it->second; 
			//TODO: check tensor order
			char* buf = New char[data_header.bytes];
			f.read(buf, data_header.bytes);
			tensor->CopyDataFromCPU(buf, data_header.bytes, data_header.data_type, data_header.dims);
			delete[]buf;
			 
			params.insert(*it);
			uninit_params.erase(it); 
		}
		else {
			cerr << "Loading parameter `" << name << "` failed. ignored.\n";
		}
		
	}
	if (uninit_params.size() > 0) {
		for (auto it = uninit_params.begin(); it != uninit_params.end(); it++) {
			it->second->Randomize();
			params.insert(*it);
		}
		uninit_params.clear();
	}

	delete[]str_buf;
	f.close();
	return true;
}
 
bool ParamPool::Save(const char * filename) {

	ofstream f(filename, ios::binary | ios::trunc );
	if (!f.is_open()) return false;
	size_t str_bytes = 0; 
	for (auto it = params.begin(); it != params.end(); it++) {
		str_bytes += (it->first.length() + 1);
	}
	size_t written = 0;
	uint8_t* str_buf = New uint8_t[str_bytes];
	uint32_t index = 0;
	for (auto it = params.begin(); it != params.end(); it++) {
		str_buf[index ++] = (uint8_t) it->first.length();
		memcpy(str_buf + index, it->first.c_str(), it->first.length());
		index += it->first.length();
	}
	//TODO : allow TO_NHWC
	param_file_header header = { 1,0, tensor_order, 0, iteration * GetAppConfig().GetBatch(), str_bytes};

	f.write(reinterpret_cast<char*>(&header), sizeof(param_file_header));
	f.write(reinterpret_cast<char*>(str_buf), str_bytes);

	tensor_data_header data_header;
	index = 0;
	for (auto it = params.begin(); it != params.end(); it++) {
		data_header.data_type = DT_FLOAT32;
		data_header.name_index = index;
		FloatTensor4D* tensor = it->second;
		data_header.dims[0] = tensor->GetBatch();
		if (tensor_order == TO_NCHW) {
			data_header.dims[1] = tensor->GetChannels();
			data_header.dims[2] = tensor->GetHeight();
			data_header.dims[3] = tensor->GetWidth();
		}
		else {			
			data_header.dims[1] = tensor->GetHeight();
			data_header.dims[2] = tensor->GetWidth();
			data_header.dims[3] = tensor->GetChannels();
		}
		data_header.bytes = tensor->MemBytes();

		f.write(reinterpret_cast<char*>(&data_header), sizeof(tensor_data_header));
		written += sizeof(tensor_data_header); 
		char* cpu_data = tensor->CopyToCPU();
		f.write(cpu_data, tensor->MemBytes());
		delete[]cpu_data;

		index += (it->first.length() + 1); 
	}
	f.close();
	return true;
}
 

FloatTensor4D * ParamPool::GetParameter(const string& key) {
	auto it = params.find(key);
	if(it != params.end()) return NULL;
	return it->second;
}
enum DarknetLayerType {
	DLT_CONVOLUTIONAL,
	DLT_MAXPOOL,
	DLT_YOLO,
	DLT_UPSAMPLE,
	DLT_ROUTE,
	DLT_SHORTCUT,

};
struct DarknetLayer {
	DarknetLayerType layer_type;
	bool batch_norm;
	bool padding;
	int filters;
	int size;
	int channels;
	int stride; 
	int layers[8];
	
	char activation[20];
	char output_id[20];
	char last_module[40];
	char anchors[40];
};
bool ParamPool::TransformDarknetWeights(const char* cfg, const char* filename, const char* out_filename) {
	fstream netfile(cfg , ios::in);
	ifstream weightsfile(filename , ios::binary);
	int layer_index = 0;
	if (!netfile.is_open() || !weightsfile.is_open()) return false;
 
	
	int headings[5];
	/*
	int major;
	int minor;
	int revision;
	int iteration int32 or unint64
	*/

	weightsfile.read((char*)headings, sizeof(int) * 4);
	iteration = (uint32_t)headings[3] / GetAppConfig().GetBatch();
	if (headings[0] * 10 + headings[1] >= 2) {
		//2 ? 20
		weightsfile.read((char*)&headings[4], sizeof(int));
	}
 
	vector<DarknetLayer> layers;
	string layer_type;
	string line; 
	release_mem = true; 
	
 
	int output_id = 1;
	int channels = 3;
	int classes = 1;
	int input_width = 416;
	int input_height = 416;
	vector<int> anchors;
	while (!netfile.eof()) {
		getline(netfile, line);
		trim(line);
		if (line.empty() || line == "[net]" || line[0] == '#') continue;
		if (netfile.eof() || (line[0] == '[' && line[line.length() - 1] == ']')) {
			layer_type = line.substr(1, line.length() - 2);
			DarknetLayer temp;
			memset(&temp, 0, sizeof(DarknetLayer));
			temp.channels = channels;
			if (layer_type == "conv" || layer_type == "convolutional") {
				temp.layer_type = DLT_CONVOLUTIONAL;
				temp.filters = 1;
				temp.size = 1;
				temp.stride = 1;
			}
			else if (layer_type == "maxpool") {
				temp.layer_type = DLT_MAXPOOL;
				temp.size = 2;
				temp.stride = 2; 
			}
			else if (layer_type == "yolo") {
				temp.layer_type = DLT_YOLO;
			}
			else if (layer_type == "route") {
				temp.layer_type = DLT_ROUTE;
				output_id--;
			}
			else if (layer_type == "upsample") {
				temp.layer_type = DLT_UPSAMPLE;
				temp.stride = 1;
				
			}
			else if (layer_type == "shortcut") {
				temp.layer_type = DLT_SHORTCUT;
			}			 
			else {
				cerr << "Warning: Unprocessed layer type `" << layer_type << "`!\n";
			}
			sprintf(temp.output_id, "layer%02d", output_id++);
			layers.push_back(temp);
		}
		else {
			size_t pos = line.find('=');
			if (string::npos == pos) continue;
			string key = line.substr(0, pos);
			string val = line.substr(pos + 1);
			trim(key);
			trim(val);

			if (0 == layers.size()) { 
				if (key == "classes") classes = atoi(val.c_str());
				else if(key == "width") input_width = atoi(val.c_str());
				else if (key == "height") input_height = atoi(val.c_str());
				else if (key == "channels") channels = atoi(val.c_str());
				continue;
			};
			
			
			DarknetLayer& currentLayer = layers[layers.size() - 1];
			if (key == "batch_normalize")
				currentLayer.batch_norm = (val == "1") || (val == "true");
			else if (key == "filters") {
				currentLayer.filters = atoi(val.c_str());
				channels = currentLayer.filters;
			}
			else if (key == "size")
				currentLayer.size = atoi(val.c_str());
			else if (key == "stride")
				currentLayer.stride = atoi(val.c_str());
			else if (key == "from") {
				int t = atoi(val.c_str());
				if (t < 0) t += (layers.size() - 1);
				currentLayer.layers[0] = t;
				if (layers[t].layer_type == DLT_CONVOLUTIONAL) {
					channels = layers[t].filters;
				}
				else {
					channels = layers[t].channels;
				}
				 
			}
			else if (key == "pad") {
				currentLayer.padding = (val == "1") || (val == "true");
			}
			else if (key == "activation") {
				strcpy(currentLayer.activation, val.c_str());
			}
			else if(key == "layers"){
				vector<string> strs;
				split_string(strs, val, ',');
				channels = 0;
				for (int i = 0; i < min(8, strs.size()); i++) {
					int t = atoi(strs[i].c_str());
					if (t < 0) t += (layers.size() - 1);
					
					if (layers[t].layer_type == DLT_CONVOLUTIONAL) {
						channels += layers[t].filters;
					}
					else {
						channels += layers[t].channels;
					}
				 
					currentLayer.layers[i] = t;					 
				}
			}
			else if (key == "mask") {
				strcpy(currentLayer.anchors, val.c_str());
			}
			else if (key == "anchors" && anchors.size() == 0) {
				vector<string> strs;
				split_string(strs, val, ',');
				for (int i = 0; i < strs.size(); i++) {
					anchors.push_back(atoi(strs[i].c_str()));
				}
			}
		} 
	}
	netfile.close();
	string cfg_new(cfg);
	replace_extension(cfg_new, ".xml");
	netfile.open(cfg_new.c_str(),ios::out | ios::trunc);
	
	if(netfile.is_open()){
		netfile << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" <<
			"<net version=\"1.0\">\n" <<
			"\t<input>\n" <<
			"\t\t<data_order>NCHW</data_order>\n" <<
			"\t\t<data_type>FP32</data_type>\n" << 
			"\t\t<channels>"<< ((layers.size() > 0) ? layers[0].channels : 3 )<<"</channels>\n" <<
			"\t\t<height>" << input_height << "</height>\n" <<
			"\t\t<width>" << input_width << "</width>\n" <<
			"\t\t<classes>" << classes << "</classes>\n" <<
			"\t</input>\n" <<
			"\t<anchors>\n"  ;
		for (int i = 0; i < anchors.size(); i += 2) {
			netfile << "\t\t<anchor width=\"" << anchors[i] << "\" height=\"" << anchors[i + 1] << "\" />\n";
		}
		netfile << "\t</anchors>\n\t<layers>\n";
		string before("none"); 
		int index = 1;
		string param_name;
		FloatTensor4D* tensor;
		for (DarknetLayer& l : layers) { 
			if(l.layer_type != DLT_ROUTE) 
				netfile << "\t\t<layer id=\"" << l.output_id << "\">\n";
			char* cpu_data = NULL;
			switch (l.layer_type) {
			case DLT_CONVOLUTIONAL:
				
				if (l.batch_norm) {
					netfile << "\t\t\t<module  type=\"conv\"  id=\"convolution\" filters=\"" << l.filters
						<< "\" filter-w=\"" << l.size << "\" filter-h=\"" << l.size
						<< "\" stride-w=\"" << l.stride << "\"  stride-h=\"" << l.stride 
						<< "\" bias=\"false\" before=\"" << before << "\"/>\n";
					netfile << "\t\t\t<module id=\"normalization\" type=\"batch-norm\" before=\"convolution\" />\n";
					before = "normalization";
					param_name = l.output_id;
					param_name += ".normalization";
					tensor = New FloatTensor4D();
					tensor->Init(4, l.filters, 1,1, TO_NCHW);
					params.insert(pair<string, FloatTensor4D*>(param_name, tensor));
					cpu_data = tensor->CopyToCPU();
					weightsfile.read(cpu_data, tensor->MemBytes());
					cout << "Load " << param_name << ": " << tensor->MemBytes() << " bytes.\n";
				}
				else {
					netfile << "\t\t\t<module  type=\"conv\"  id=\"convolution\" filters=\"" << l.filters
						<< "\" filter-w=\"" << l.size << "\" filter-h=\"" << l.size
						<< "\" stride-w=\"" << l.stride << "\"  stride-h=\"" << l.stride 
						<< "\" bias=\"true\" before=\"" << before << "\"/>\n";
					before = "convolution";
					param_name = l.output_id;
					param_name += ".convolution.bias";
					tensor = New FloatTensor4D();
					tensor->Init(1, l.filters, 1, 1, TO_NCHW);
					params.insert(pair<string, FloatTensor4D*>(param_name, tensor));
					cpu_data = tensor->CopyToCPU();
					weightsfile.read(cpu_data, tensor->MemBytes());
					cout << "Load " << param_name << ": " << tensor->MemBytes() << " bytes.\n";
				}
				param_name = l.output_id;
				param_name += ".convolution.weights";
				tensor = New FloatTensor4D();
				
				tensor->Init(l.filters, l.channels, l.size, l.size, TO_NCHW);
				if (param_name == "layer20.weights") {
					cout << " dim [" << tensor->GetBatch() << ", " << tensor->GetChannels() << ", " << tensor->GetHeight()
						<< ", " << tensor->GetWidth() << "]\n";
				}
				cpu_data = tensor->CopyToCPU();
				weightsfile.read(cpu_data, tensor->MemBytes());
				cout << "Load " << param_name << ": " << tensor->MemBytes() << " bytes.\n";
				params.insert(pair<string,FloatTensor4D*>(param_name, tensor));
				netfile << "\t\t\t<module id=\"activation\" type=\"activation\" method=\"" << l.activation 
					<< "\" before=\"" << before << "\" />\n";
				sprintf(l.last_module, "%s.activation", l.output_id);
				before = l.last_module;
				break;
			case DLT_MAXPOOL:
				if (0 == l.size) l.size = l.stride;
				
				netfile << "\t\t\t<module id=\"pool\" type=\"max-pool\" window-w=\"" << l.size
					<< "\" window-h=\"" << l.size << "\" stride-w=\"" << l.stride <<
					"\" stride-h=\"" << l.stride << "\"  before=\"" << before << "\" />\n";

				sprintf(l.last_module, "%s.pool", l.output_id);
				before = l.last_module; 
				break;
			case DLT_SHORTCUT: 
				before = layers[l.layers[0]].last_module; 
				netfile << "\t\t\t<module id=\"activation\" type=\"activation\" method=\"" 
					<< l.activation << "\" before=\"" << before << "\" />\n";
				sprintf(l.last_module, "%s.activation", l.output_id);
				before = l.last_module; 
				break;
			case DLT_ROUTE:
				before = ""; 
				for (int i = 0; i < 8 && l.layers[i] != 0; i++) {
					if (i > 0) before += ','; 
					before += layers[l.layers[i]].last_module;
					
				}
				break;
			case DLT_UPSAMPLE:
				netfile << "\t\t\t<module id=\"upsample\" type=\"upsample\"  stride-w=\"" << l.stride <<
					"\" stride-h=\"" << l.stride << "\"  before=\"" << before << "\" />\n";

				sprintf(l.last_module, "%s.upsample", l.output_id);
				before = l.last_module;
				break;
			case DLT_YOLO:
				netfile << "\t\t\t<module id=\"yolo\" type=\"yolo-detection\" before=\"" << before 
					<< "\" ignore-thresh=\"0.7\" truth-thresh=\"1.0\" anchor-masks=\"" << l.anchors << "\" />\n";
				before = "";
			}
			if (cpu_data) {
				delete[]cpu_data;
				cpu_data = NULL;
			}
			if (l.layer_type != DLT_ROUTE)
				netfile << "\t\t</layer>\n";

		}
		netfile << "\t</layers>\n</net>\n";
	}
	netfile.close(); 
	weightsfile.close();
	bool r;
	if(out_filename != NULL && 0 != *out_filename )
		 r = Save(out_filename);
	else {
		string outname(filename);
		replace_extension(outname, ".rweights");
		r = Save(outname.c_str());
	}
	
	for (auto e : params) {
		delete e.second;
	}
	params.clear();
	return r;
}

 