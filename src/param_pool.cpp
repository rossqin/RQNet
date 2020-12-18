#include "stdafx.h"
#include "cuda_tensor.h"
#include "param_pool.h"
#include "config.h"

ParamPool::~ParamPool() {
	if (release_mem) {
		for (auto e : params) {
			delete e.second;
		}
	}
}

void ParamPool::Put(string key, CudaTensor* tensor) {
	params.insert(pair<string, CudaTensor*>(key, tensor));
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
	size_t total_read = sizeof(param_file_header) + f.gcount();
	
	while (!f.eof()) {
		f.read(reinterpret_cast<char*>(&data_header), sizeof(tensor_data_header));
		size_t read_length = f.gcount();
		if (0 == read_length) {
			// f.eof() is false , really strannge
			break;
		}
		name.assign(str_buf + data_header.name_index + 1, static_cast<uint8_t>(str_buf[data_header.name_index]));
		if (read_length < sizeof(tensor_data_header)) {
			cerr << " Warning: Some params are not loaded from weights file!\n";
			break;
		}
		//TODO: check datatype
		total_read += read_length;
 
		CpuPtr<char> buf(data_header.bytes);
		f.read(buf.ptr, data_header.bytes);
		read_length = f.gcount();
		total_read += read_length;
		auto it = params.find(name);
		if (it != params.end()) { 
			if (CUDNN_DATA_FLOAT == data_header.data_type) {
				float* f = reinterpret_cast<float*>(buf.ptr);
				int num = data_header.bytes >> 2;
				for (int i = 0; i < num; i++) {
					//if (first_conv) f[i] /= 1024.0f;
					if (f[i] > -1.0e-10f && f[i] < 1.0e-10f)
						f[i] = 0.0f;
				}
			}
			if (!it->second->Push(buf, data_header)) {
				cerr << " INFO: Parameter `" << name << "` loaded but not set. Ignore.\n";
			}
		}
		else {			 
			cerr << " INFO: Parameter `" << name << "` not in network. Ignore.\n";
		} 
	}  
	delete[]str_buf;
	f.close();
	return true;
}
 
bool ParamPool::Save(const char * filename, int i) {

	ofstream f(filename, ios::binary | ios::trunc );
	if (!f.is_open()) return false;
	size_t str_bytes = 0; 
	for (auto it = params.begin(); it != params.end(); it++) {
		str_bytes += (it->first.length() + 1);
	}
	size_t written = 0;
	uint8_t* str_buf = New uint8_t[str_bytes];
	size_t index = 0;
	for (auto it = params.begin(); it != params.end(); it++) {
		str_buf[index ++] = (uint8_t) it->first.length();
		memcpy(str_buf + index, it->first.c_str(), it->first.length());
		index += it->first.length();
	}
	if(i > 0) iteration = i;
	//TODO : allow TO_NHWC
	param_file_header header = { 1,0, tensor_order, 0, iteration * GetAppConfig().GetBatch(), str_bytes};

	f.write(reinterpret_cast<char*>(&header), sizeof(param_file_header));
	f.write(reinterpret_cast<char*>(str_buf), str_bytes);
	delete[]str_buf;

	tensor_data_header data_header;
	index = 0;
	for (auto it = params.begin(); it != params.end(); it++) {
		data_header.data_type = CUDNN_DATA_FLOAT;
		data_header.name_index = index;
		CudaTensor* tensor = it->second;
		data_header.dims[0] = tensor->Batch();
		if (tensor->DataFormat() == CUDNN_TENSOR_NCHW) {
			data_header.dims[1] = tensor->Channel();
			data_header.dims[2] = tensor->Height();
			data_header.dims[3] = tensor->Width();
		}
		else {			
			data_header.dims[1] = tensor->Height();
			data_header.dims[2] = tensor->Width();
			data_header.dims[3] = tensor->Channel();
		}
		data_header.bytes = tensor->Bytes();

		f.write(reinterpret_cast<char*>(&data_header), sizeof(tensor_data_header));
		written += sizeof(tensor_data_header); 

		char* cpu_data = New char[tensor->Bytes()];
		cudaMemcpy(cpu_data, tensor->Data(), tensor->Bytes(),cudaMemcpyDeviceToHost);
		f.write(cpu_data, tensor->Bytes());
		delete[]cpu_data;

		index += (it->first.length() + 1); 
	}
	f.close();
	return true;
}
 

CudaTensor * ParamPool::GetParameter(const string& key) {
	auto it = params.find(key);
	if(it != params.end()) return nullptr;
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

	bool focal_loss; 
	
	char activation[20];
	char output_id[20];
	char last_module[40];
	char anchors[40];
};
bool ParamPool::TransformDarknetWeights(const char* cfg, const char* filename, const char* out_dir) {
	fstream netfile(cfg , ios::in);
	ifstream weightsfile(filename , ios::binary);
	int layer_index = 0;
	if (!netfile.is_open() || !weightsfile.is_open()) return false;
 
	
	int headings[5];


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
				temp.focal_loss = true;
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
				currentLayer.layers[0] = layers.size() - 2;
				if (t < 0) t += (layers.size() - 1);
				currentLayer.layers[1] = t;
			/*
				if (layers[t].layer_type == DLT_CONVOLUTIONAL) {
					channels = layers[t].filters;
				}
				else {
					channels = layers[t].channels;
				}
				 */
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
				int size = strs.size();
				if (size > 8) size = 8;
				for (int i = 0; i < size; i++) {
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
			else if (key == "classes") {
				classes = atoi(val.c_str());
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
			else if (key == "focal_loss") {
				currentLayer.focal_loss = (val == "1") || (val == "true");
			}
		} 
	}
	netfile.close();
	char fname[MAX_PATH];
	_splitpath(cfg, nullptr, nullptr, fname, nullptr);
	string dest_dir(out_dir);
	if (dest_dir.empty())
		dest_dir = "./";
	else if (dest_dir.at(dest_dir.length() - 1) != '/' && dest_dir.at(dest_dir.length() - 1) != '\\')
		dest_dir += SPLIT_CHAR;
	 
	netfile.open((dest_dir + fname + ".xml").c_str(),ios::out | ios::trunc);
	
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
		CudaTensor *tensor;
		float* cpu_data;
		for (DarknetLayer& l : layers) { 
			if(l.layer_type != DLT_ROUTE) 
				netfile << "\t\t<layer id=\"" << l.output_id << "\">\n"; 
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
					tensor = New CudaTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);
					tensor->Init({ 4, l.filters, 1,1 });
					params.insert(pair<string, CudaTensor*>(param_name, tensor));
					cpu_data = New float[tensor->Elements()];
					weightsfile.read(reinterpret_cast<char*>(cpu_data), tensor->Bytes());
					cout << "Load " << param_name << ": " << tensor->Bytes() << " bytes.\n";
					tensor->Push(cpu_data);
					delete[]cpu_data;
				}
				else {
					netfile << "\t\t\t<module  type=\"conv\"  id=\"convolution\" filters=\"" << l.filters
						<< "\" filter-w=\"" << l.size << "\" filter-h=\"" << l.size
						<< "\" stride-w=\"" << l.stride << "\"  stride-h=\"" << l.stride 
						<< "\" bias=\"true\" before=\"" << before << "\"/>\n";
					before = "convolution";
					param_name = l.output_id;
					param_name += ".convolution.bias";
					CudaTensor *tensor = New CudaTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);
					tensor->Init({ 1, l.filters, 1, 1 });
					params.insert(pair<string, CudaTensor*>(param_name, tensor));
					cpu_data = New float[tensor->Elements()];
					weightsfile.read(reinterpret_cast<char*>(cpu_data), tensor->Bytes());
					cout << "Load " << param_name << ": " << tensor->Bytes() << " bytes.\n";
					tensor->Push(cpu_data);
					delete[]cpu_data;
				}
				param_name = l.output_id;
				param_name += ".convolution.weights";
				tensor = New CudaTensor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);				
				tensor->Init({ l.filters, l.channels, l.size, l.size });
				if (param_name == "layer20.weights") {
					cout << " dim [" << tensor->Batch() << ", " << tensor->Channel() << ", " << tensor->Height()
						<< ", " << tensor->Width() << "]\n";
				} 
				cpu_data = New float[tensor->Elements()];
				weightsfile.read(reinterpret_cast<char*>(cpu_data), tensor->Bytes());
				cout << "Load " << param_name << ": " << tensor->Bytes() << " bytes.\n";
				tensor->Push(cpu_data);
				delete[]cpu_data;
				params.insert(pair<string, CudaTensor*>(param_name, tensor)); 
				if (_strcmpi(l.activation, "linear")) {
					netfile << "\t\t\t<module id=\"activation\" type=\"activation\" method=\"" << l.activation
						<< "\" before=\"" << before << "\" />\n";
					sprintf(l.last_module, "%s.activation", l.output_id);
				}
				else
					sprintf(l.last_module, "%s.convolution", l.output_id);
				before = l.last_module;
				break;
			case DLT_MAXPOOL:
				if (0 == l.size) l.size = l.stride;
				
				netfile << "\t\t\t<module id=\"pool\" type=\"max-pool\" window-w=\"" << l.size
					<< "\" window-h=\"" << l.size << "\" stride-w=\"" << l.stride <<
					"\" stride-h=\"" << l.stride << "\" ";

				if (l.stride == 1) {
					netfile << "pads_begin=\"0,0\" pads_end=\"1,1\" ";
				}
				
				netfile << "before=\"" << before << "\" />\n";

				sprintf(l.last_module, "%s.pool", l.output_id);
				before = l.last_module; 
				break;
			case DLT_SHORTCUT: 
				before = "";
				for (int i = 0; i < 8 && l.layers[i] != 0; i++) {
					if (i > 0) before += ',';
					before += layers[l.layers[i]].last_module;

				}
				netfile << "\t\t\t<module id=\"shortcut\" type=\"shortcut\" before=\"" << before << "\" />\n";
				
				if (0 == _strcmpi(l.activation, "linear")) {
					sprintf(l.last_module, "%s.shortcut", l.output_id);
				}
				else {
					netfile << "\t\t\t<module id=\"activation\" type=\"activation\" method=\"" << l.activation
						<< "\" before=\"shortcut\" />\n";
					sprintf(l.last_module, "%s.activation", l.output_id);
				}
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
					<< "\" ignore-thresh=\"0.7\" truth-thresh=\"1.0\" anchor-masks=\"" << l.anchors 
					<< "\" focal-loss=\""<< (l.focal_loss ? "true" : "false") <<"\" />\n";
				before = "";
			} 
			if (l.layer_type != DLT_ROUTE)
				netfile << "\t\t</layer>\n";

		}
		netfile << "\t</layers>\n</net>\n";
		cout << "\n INFO - Network definition file `"  << dest_dir <<  fname << ".xml` written successfully!\n";
		netfile.close();
	}
	
	weightsfile.close();
	  
	bool r = Save((dest_dir + fname + ".rweights").c_str());
	if (r) {
		cout << "\n INFO -  Weights file `" << dest_dir << fname << ".rweights written successfully!\n";
	}
	else
		cerr << " Error: file `" << dest_dir << fname << ".rweights written failed!\n";
 
	
	for (auto e : params) {
		delete e.second;
	}
	params.clear();
	return r;
}

bool ParamPool::DumpAsExcels(const char* output_dir) const {

	if (!make_sure_dir_exists(output_dir)) {
		cerr << "Failed to create folder `" << output_dir << "` !\n";
		return false;
	}
	string outputdir_with_slash(output_dir);
	if (outputdir_with_slash.find_last_of('/') != outputdir_with_slash.length() &&
		outputdir_with_slash.find_last_of('\\') != outputdir_with_slash.length())
		outputdir_with_slash += SPLIT_CHAR;
 
	for (auto it = params.begin(); it != params.end(); it++) {
		CudaTensor* tensor = it->second;
		cudnnDataType_t fmt = tensor->DataType();
		if (fmt != CUDNN_DATA_FLOAT && fmt != CUDNN_DATA_HALF ) {
			cerr << "Only Support CUDNN_DATA_FLOAT and CUDNN_DATA_HALF! \n";
			return false;
		} 		
		CpuPtr<char> ptr(tensor->Bytes(), tensor->Data());
		float* dataf = reinterpret_cast<float*>(ptr.ptr);
		half* datah = reinterpret_cast<half*>(ptr.ptr);
		char full_path[MAX_PATH];
		char buff[20];
		string cell_content;
		sprintf_s(full_path, MAX_PATH, "%s%s.csv", outputdir_with_slash.c_str(), it->first.c_str());
		//BasicExcel wb;
		//wb.Create(1);
		ofstream csvf;
		csvf.open(full_path, ios::out | ios::trunc);
		//BasicExcelWorksheet* sheet = wb.GetWorksheet((size_t)0);
		for (int b = 0; b < tensor->Batch(); b++) { 	
			
			for (int i = 0; i < tensor->Channel(); i++) { 
				cell_content = "\"";
				for (int j = 0; j < tensor->Height(); j++) {
					for (int k = 0; k < tensor->Width(); k++) {
						if (CUDNN_DATA_FLOAT == fmt) {
							sprintf_s(buff, 20, "%.6f", *dataf);
							//sheet->Cell(i * tensor->Height() + j, k)->SetDouble();
							dataf++;
						}
						else {
							sprintf_s(buff, 20, "%.6f", (float)*datah);
							//sheet->Cell(i * tensor->Height() + j, k)->SetDouble(*datah);
							datah ++;
						}
						cell_content += buff;
						if (k < tensor->Width() - 1) cell_content += ",";
					}
					if (j < tensor->Height() - 1) cell_content += "\n";
				}
				cell_content += "\",";
				csvf.write(cell_content.c_str(), cell_content.length());

				//sheet->Cell(b, i)->SetString(cell_content.c_str());				
			}
			csvf.write("\n", 1);
		}
		csvf.close();
		cout << "Save excel file to `" << full_path << "`!\n";
		
	}
	return true;
}
void ParamPool::Reset() {
	params.clear();
}