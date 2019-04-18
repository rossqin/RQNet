#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"
#include "param_pool.h"
#include "image.h"
static CNNNetwork network;
CNNNetwork& GetNetwork() {
	return network;
}

ModulePool& GetModulePool() {
	return network.module_pool;
}
CNNNetwork::CNNNetwork() {
	data_type = CUDNN_DATA_FLOAT; 
	data_format = CUDNN_TENSOR_NCHW;
	workspace = nullptr;
	workspace_size = 0;
	def_actvation = "leaky";
	truths = nullptr;
	input = nullptr;
}

CNNNetwork::~CNNNetwork() {
	for (auto l : layers) {
		delete l;
	}
	if (workspace) cudaFree(workspace);
	for (auto m = module_pool.begin(); m != module_pool.end(); m++) {
		delete m->second;
	} 
	if (truths) delete[]truths;
	if (input) delete[] input;
}
static cudnnDataType_t get_data_type(const string& str) {
	if (str == "FP32") {
		return CUDNN_DATA_FLOAT;
	}
	else if(str == "FP16"){
		return CUDNN_DATA_HALF;
	}
 
	cerr << "Error: data type `" << str << "` has not been supported yet. set to FP32.\n";
	return CUDNN_DATA_FLOAT;
	 
}
 
bool CNNNetwork::Load(const char* filename) {
	tinyxml2::XMLDocument doc;
	if (XML_SUCCESS != doc.LoadFile(filename)) return false;
	mini_batch = GetAppConfig().GetBatch() / GetAppConfig().GetSubdivision();

	XMLElement* root = doc.RootElement();
	if (!root) return false;
	root->QueryText("def-activation", def_actvation); 
	XMLElement* inputElement = root->FirstChildElement("input");
	if (inputElement) {
		string str;
		inputElement->QueryText("data_order", str);
		if (str == "NCHW")
			data_format = CUDNN_TENSOR_NCHW;
		else if (str == "NHWC")
			data_format = CUDNN_TENSOR_NHWC;
		else {
			cerr << "Error: `" << str << "` is not a valid data order. ignore, set to NCHW.\n";
			data_format = CUDNN_TENSOR_NCHW;
		}
		str = "";
		inputElement->QueryText("data_type", str);
		data_type = get_data_type(str);
		input_channels = 3;
		input_width = 416;
		input_height = 416;
		inputElement->QueryIntText("channels", input_channels);
		inputElement->QueryIntText("width", input_width);
		inputElement->QueryIntText("height", input_height);
		input = New float[mini_batch * input_channels * input_width * input_height];
	}
	XMLElement* anchorElement = root->FirstChildElement("anchors/anchor");
	while (anchorElement) {
		float w = anchorElement->FloatAttribute("width", 0.0f) / 416.0f;
		float h = anchorElement->FloatAttribute("height", 0.0f) / 416.0f;
		if (w > 0.0 && h > 0.0) {			
			anchors.push_back(pair<float, float>(w, h ));
		}
		anchorElement = anchorElement->NextSiblingElement();
	}
	XMLElement* layerElement = root->FirstChildElement("layers/layer");
	int index = 0;
	InferenceModule* last_module = NULL;
	while (layerElement) {
		Layer* layer = New Layer(layerElement,++index,this, last_module);
		layers.push_back(layer);
		layerElement = layerElement->NextSiblingElement();
	}
	truths = New ObjectInfo[mini_batch * GetAppConfig().GetMaxTruths()];
	return true;
}

Layer * CNNNetwork::GetLayer(int index) const {
	if (index < 0 || index >(int)layers.size()) return NULL;
	return layers[index]; 
}

bool CNNNetwork::GetAnchor(int index, float & width, float & height) {
	if (index < 0 || index >(int)anchors.size()) return false;
	width = anchors[index].first;
	height = anchors[index].second;
	return true;
}

bool CNNNetwork::UpdateWorkspace(size_t new_size) {
	if (new_size > workspace_size) {
		workspace_size = new_size;
		cudaFree(workspace);
		workspace = NULL;
		return cudaSuccess == cudaMalloc(&workspace, workspace_size);
	}
	return true;
}
bool CNNNetwork::Forward(bool training) {
	loss = 0.0;
	ForwardContext context = { training,
		GetAppConfig().ConvParamsFreezed(),
		GetAppConfig().BNParamsFreezed(),
		GetAppConfig().ActParamsFreezed(),
		nullptr,
		GetAppConfig().GetMaxTruths(),
		truths };
	for (int i = 0; i < (int)layers.size(); i++) {
		Layer* l = layers[i];
		if (!l->Forward(context)) {
			cerr << "Error: " << l->GetName() << " forward failed!\n";
			return false;
		}	 
	} 
	return true;
} 
bool CNNNetwork::Backward() {
	CudaTensor d(data_type,data_format); 
	for (int i = (int)layers.size() - 1; i >= 0; i--) { 
		Layer* l = layers[i];
		if (!l->Backward(d)) {
			cerr << "Error: " << l->GetName() << " backwawd failed!\n";
			return false;
		} 
	}
	return true;
}

bool CNNNetwork::Train() { 
	if (!cuDNNInitialize()) {
		cerr << "CUDNN initialization failed!\n";
		return false;
	}
	
	DataLoader loader;

	string filename("loss_");
	filename += get_time_str();
	filename += ",log";

	uint32_t it;
	if (GetAppConfig().FromFirstIteration())
		it = 0;
	else
		it = GetParamPool().GetIteration();

	cout << "\n *** Start training from iteration " << (it + 1) << "...\n\n";
	int new_width, new_height;
	string weights_file;
	float avg_loss = -1.0;  
	int input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
	size_t truth_len = mini_batch * GetAppConfig().GetMaxTruths() * sizeof(ObjectInfo);
#if 0
 
	ifstream f(DEBUGGING_DIR "input01.bin", ios::binary);
	f.read(reinterpret_cast<char*>(input), input_len);
	f.close();
	f.open(DEBUGGING_DIR "truth01.bin", ios::binary);
	f.read(reinterpret_cast<char*>(truths), truth_len);
	f.close();
#endif
	while (!GetAppConfig().IsLastIteration(it)) {
		loss = 0.0;
		it++;
		clock_t start_clk = clock(); 		
		for (int i = 0; i < GetAppConfig().GetSubdivision(); i++) { 
			

			//cout << "\nSubdivision " << i << ": loading data ... ";
			long t = GetTickCount();

			memset(truths, 0, truth_len);
			memset(input, 0,  input_len);
			if (!loader.MiniBatchLoad(input, truths, input_channels, mini_batch, input_width, input_height)) {
				return false;
			} 
		
#if 0
			t = GetTickCount() - t;

			char dbg_filename[MAX_PATH];
			sprintf(dbg_filename, DEBUGGING_DIR "input%02d.bin", i); 
			ofstream f(dbg_filename, ios::binary);
			f.write(reinterpret_cast<char*>(input), input_len);
			f.close();
			sprintf(dbg_filename, DEBUGGING_DIR "truth%02d.bin", i); 
			f.open(dbg_filename, ios::binary);
			f.write(reinterpret_cast<char*>(truths), truth_len);
			f.close(); 
#endif			
			if (!Forward(true)) return false; 
			if (!Backward()) return false;   
		}
		
		loss /=  GetAppConfig().GetBatch();
		if (avg_loss < 0)
			avg_loss = loss;
		else
			avg_loss = avg_loss * 0.9 + loss * 0.1;
		int ms = (clock() - start_clk) * 1000 / CLOCKS_PER_SEC;
		float lr = GetAppConfig().GetCurrentLearningRate(it);
		cout << "\n >> It " << it << " | Loss: " << loss <<", Avg-Loss: "<< avg_loss 
			<<", Learn-Rate: " << lr << ", Time: " << (ms * 0.001) << "s, Images: "
			<< it * GetAppConfig().GetBatch() << ".\n\n";
		 
		ofstream ofs(filename,ios::app);
		if (ofs.is_open()) {
			ofs << setw(10) << it << ", " << input_width  << ", " << input_height  << ", " << setw(10) << lr << ", " << setw(10) << loss << ", " << setw(10) << avg_loss << endl;
			ofs.close();
		}
		
		for (auto l : layers) {
			if (!l->Update(lr)) return false;
		} 
		if (GetAppConfig().GetWeightsPath(it, weights_file)) {
			GetParamPool().Save(weights_file.c_str(),it);
		} 
		if (GetAppConfig().RadmonScale(it, new_width, new_height) &&
			(new_width != input_width || new_height != input_width) ) {
			cout << "Input Resizing to " << new_width << "x" << new_height << " ...\n";
			cudaError_t err = cudaPeekAtLastError();
			if (err != cudaSuccess) {
				cerr << "cuda Error:" << (int)err << endl;
			}
			input_width = new_width;
			input_height = new_height;
			input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
			input = (float*)realloc(input, input_len);

		} 
		
	}
	//TODO : save final 
	return true;
}

bool CNNNetwork::OutputIRModel(const string & dir, const string & name, bool fp16) const {
	string prefix = dir;
	if (prefix.find_first_of('\\') != prefix.length() - 1 &&
		prefix.find_first_of('/') != prefix.length() - 1)
		prefix += '/';
	//TODO: make sure dir exists
	prefix += name;

	ofstream  xml(prefix + ".xml", ios::trunc);
	ofstream  bin(prefix + ".bin", ios::binary | ios::trunc);
	if ((!xml.is_open()) || (!bin.is_open()) ) return false;

	xml << "<?xml version=\"1.0\" ?>" << endl;
	//TODO: replace " with \" in name 
	xml << "<net batch=\"1\" name=\""<<name<<"\" version=\"4\">" << endl;
	xml << "  <layers>" << endl;
	stringstream edges;
	size_t bin_offset = 0;
 
	xml << "    <layer id=\"0\" name=\"inputs\" precision=\""<< (fp16 ? "FP16" : "FP32")<<"\" type=\"Input\">" << endl;
	xml << "        <output>" << endl;
	xml << "          <port id = \"0\">" << endl;
	xml << "            <dim>" << 1 << "</dim>" << endl;
	xml << "            <dim>" << input_channels << "</dim>" << endl;
	xml << "            <dim>" << input_height << "</dim>" << endl;
	xml << "            <dim>" << input_width << "</dim>" << endl;
	xml << "          </port>" << endl;
	xml << "        </output>" << endl;
	xml << "    </layer>" << endl;
 
	for (size_t i = 0; i < layers.size(); i++) {
		if (!(layers[i]->OutputIRModel(xml, bin, edges, bin_offset, fp16))) {
			return false;
		}
	}

	xml << "  </layers>" << endl;	
	xml << "  <edges>" << endl;

	xml << edges.str();

	xml << "  </edges>" << endl;
	xml << "  <meta_data>" << endl;
	//TODO: write meta data
	xml << "  </meta_data>" << endl;
	xml << "</net>" << endl;
	xml.close();
	bin.close();

	return true;
}
