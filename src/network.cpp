#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"
#include "param_pool.h"

static CNNNetwork network;
CNNNetwork& GetNetwork() {
	return network;
}

ModulePool& GetModulePool() {
	return network.module_pool;
}
CNNNetwork::CNNNetwork() {
 
	data_order = TO_NCHW;
	data_type = DT_FLOAT32; 
	workspace = NULL;
	workspace_size = 0;
	def_actvation = "leaky";
	truths = NULL;
}

CNNNetwork::~CNNNetwork() {
	for (auto l : layers) {
		delete l;
	}
	if (workspace) cudaFree(workspace);
	for (auto m = module_pool.begin(); m != module_pool.end(); m++) {
		delete m->second;
	} 
	if (truths)
		delete[]truths;
}
static DataType get_data_type(const string& str) {
	if (str == "FP32") {
		return DT_FLOAT32;
	}
	else if(str == "FP16"){
		return DT_UINT16;
	}
 
	cerr << "Error: data type `" << str << "` has not been supported yet. set to FP32.\n";
	return DT_FLOAT32;
	 
}
 
bool CNNNetwork::Load(const char* filename) {
	tinyxml2::XMLDocument doc;
	if (XML_SUCCESS != doc.LoadFile(filename)) return false;

	XMLElement* root = doc.RootElement();
	if (!root) return false;
	root->QueryText("def-activation", def_actvation); 
	XMLElement* inputElement = root->FirstChildElement("input");
	if (inputElement) {
		string str;
		inputElement->QueryText("data_order", str);
		if (str == "NCHW")
			data_order = TO_NCHW;
		else if (str == "NHWC")
			data_order = TO_NHWC;
		else {
			cerr << "Error: `" << str << "` is not a valid data order. ignore, set to NCHW.\n";
			data_order = TO_NCHW;
		}
		str = "";
		inputElement->QueryText("data_type", str);
		data_type = get_data_type(str);
		int input_channels = 3;
		int input_width = 416;
		int input_height = 416;
		inputElement->QueryIntText("channels", input_channels);
		inputElement->QueryIntText("width", input_width);
		inputElement->QueryIntText("height", input_height);
		if(!input.Init(GetAppConfig().GetMiniBatch(), input_channels, input_width, input_height, data_order))
			return false;
	}
	XMLElement* anchorElement = root->FirstChildElement("anchors/anchor");
	while (anchorElement) {
		int w = anchorElement->FloatAttribute("width", 0);
		int h = anchorElement->FloatAttribute("height", 0);
		if (w > 0 && h > 0)
			anchors.push_back(pair<float, float>(w / 416.0, h / 416.0));
		anchorElement = anchorElement->NextSiblingElement();
	}
	XMLElement* layerElement = root->FirstChildElement("layers/layer");
	int index = 0;
	InferenceModule* last_module = NULL;
	while (layerElement) {
		Layer* layer = New Layer(layerElement,++index, last_module);
		layers.push_back(layer);
		layerElement = layerElement->NextSiblingElement();
	}
	truths = New ObjectInfo[GetAppConfig().GetMiniBatch() * GetAppConfig().GetMaxTruths()];
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
		input,
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
	FloatTensor4D d;
	char filename[MAX_PATH];
	for (int i = layers.size() - 1; i >= 0; i--) { 
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
 
	while (!GetAppConfig().IsLastIteration(it)) {
		loss = 0.0;
		it++;
		clock_t start_clk = clock(); 		
		for (int i = 0; i < GetAppConfig().GetSubdivision(); i++) { 
 
			//cout << "\nSubdivision " << i << ": loading data ... ";
			long t = GetTickCount();
			input = 0.0f;
			size_t len = GetAppConfig().GetMiniBatch() * GetAppConfig().GetMaxTruths() * sizeof(ObjectInfo);
			memset(truths, 0, len);
			input = 0.0f; 
			if (!loader.MiniBatchLoad(input, truths)) return false;  
			t = GetTickCount() - t;
			//cout << " in " << (t * 0.001) << " secs.\n";
			
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
			ofs << setw(10) << it << ", " <<  input.GetWidth()<< ", " << input.GetHeight()<< ", " << setw(10) << lr << ", " << setw(10) << loss << ", " << setw(10) << avg_loss << endl;
			ofs.close();
		}
		
		lr /= GetAppConfig().GetBatch();
		for (auto l : layers) {
			if (!l->Update(lr)) return false;
		} 
		if (GetAppConfig().GetWeightsPath(it, weights_file)) {
			GetParamPool().Save(weights_file.c_str(),it);
		} 
		if (GetAppConfig().RadmonScale(it, new_width, new_height) &&
			(new_width != input.GetWidth() || new_height != input.GetHeight()) ) {
			cout << "Input Resizing to " << new_width << "x" << new_height << " ...\n";
			cudaError_t err = cudaPeekAtLastError();
			if (err != cudaSuccess) {
				cerr << "cuda Error:" << (int)err << endl;
			}
			if (!input.Init(GetAppConfig().GetMiniBatch(), input.GetChannels(),
				new_width, new_height, input.GetOrder())) {
				return false;
			}
		} 
		
	}
	//TODO : save final 
	return true;
}