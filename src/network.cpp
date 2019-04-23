#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"
#include "param_pool.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image.h"
#include "box.h"
CNNNetwork network;
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
	if (truths) {
		for (int i = 0; i < mini_batch; i++) {
			delete truths[i];
		}
		delete[]truths;
	}
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
 
void CNNNetwork::AddDetectionResult(const DetectionResult & data) {
	int index = (int)detections.size() - 1; 
	Box box(data.x, data.y, data.w, data.h );
	while (index >= 0) {
		DetectionResult &dr = detections[index];
		Box box2(dr.x, dr.y, dr.w, dr.h);
		if (BoxIoU(box, box2) > 0.7f) {
			if (data.confidence < dr.confidence) {
				return;
			}
			dr = data;
			return; 
		}
		index--;
	}
	detections.push_back(data);

}

bool CNNNetwork::Load(const char* filename, cudnnDataType_t dt) {
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
		if (dt != CUDNN_DATA_HALF && dt != CUDNN_DATA_FLOAT) {

			inputElement->QueryText("data_type", str);
			upper(str);
			data_type = get_data_type(str);
		}
		else
			data_type = dt;

		if(data_type == CUDNN_DATA_HALF) 
			cout << "\n *** Data type: FP16\n";
		else
			cout << "\n *** Data type: FP32\n";
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
	InferenceModule* last_module = nullptr;
	while (layerElement) {
		Layer* layer = New Layer(layerElement,++index,this, last_module);
		layers.push_back(layer);
		layerElement = layerElement->NextSiblingElement();
	}
	truths = New LPObjectInfos[mini_batch];
	for (int i = 0; i < mini_batch; i++) {
		truths[i] = New vector<ObjectInfo>();
	}
	return true;
}

Layer * CNNNetwork::GetLayer(int index) const {
	if (index < 0 || index >(int)layers.size()) return nullptr;
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
		workspace = nullptr;
		return cudaSuccess == cudaMalloc(&workspace, workspace_size);
	}
	return true;
}
bool CNNNetwork::Forward(bool training) {
	ForwardContext context = { training,
		GetAppConfig().ConvParamsFreezed(),
		GetAppConfig().BNParamsFreezed(),
		GetAppConfig().ActParamsFreezed(),
		nullptr
	};
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
	 
	while (!GetAppConfig().IsLastIteration(it)) {
		loss = 0.0f;
		training_batch = 0;
		true_positives = 0.0f;
		false_negatives = 0.0f;
		it++;
		clock_t start_clk = clock(); 		
		for (int i = 0; i < GetAppConfig().GetSubdivision(); i++) { 
			

			//cout << "\nSubdivision " << i << ": loading data ... ";
			long t = GetTickCount(); 
			memset(input, 0,  input_len);
			if (!loader.MiniBatchLoad(input, truths, input_channels, mini_batch, input_width, input_height)) {
				return false;
			} 
				
			if (!Forward(true)) return false; 
			if (!Backward()) return false;   
		}
		
		loss /= training_batch;
		true_positives /= training_batch;
		false_negatives /= training_batch;
		if (avg_loss < 0)
			avg_loss = loss;
		else
			avg_loss = avg_loss * 0.9 + loss * 0.1;
		int ms = (clock() - start_clk) * 1000 / CLOCKS_PER_SEC;
		float lr = GetAppConfig().GetCurrentLearningRate(it);
		cout << "\n >> It " << it << " | Loss: " << loss <<", Avg-Loss: "<< avg_loss 
			<<", TP: "<< true_positives <<"%, FN: "<< false_negatives  <<"%, Learn-Rate: " << lr << ", Time: " << (ms * 0.001) << "s, Images: "
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
			input_width = new_width;
			input_height = new_height;
			input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
			input = (float*)realloc(input, input_len);

		} 
		
	}
	//TODO : save final 
	return true;
}

bool CNNNetwork::OutputIRModel(const string & dir, const string & name, int& l_index) const {
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
 
	xml << "    <layer id=\"0\" name=\"inputs\" precision=\""<< Precision() <<"\" type=\"Input\">" << endl;
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
		Layer* l = layers[i];
		if (!l->FuseBatchNormModule()) {
			cerr << " Error: FuseBatchNorm failed in  " << l->GetName() << "!\n"; 
			return false;
		}
		if (!(l->OutputIRModel(xml, bin, edges, bin_offset, l_index))) {
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
void CNNNetwork::GetAnchorsStr(string & str) const {
	str = "";
	for (int i = 0; i < (int)anchors.size(); i++) {
		auto& a = anchors[i];
		int w = (int)(a.first * 416.0f);
		int h = (int)(a.second * 416.0f);
		str += to_string(w) + "," + to_string(h);
		if (i != (int)anchors.size() - 1) {
			str += ",";
		}
	}
}
void DrawTruthInGrid(const cv::Mat& orig_img, int stride, const cv::Scalar& color, const vector<ObjectInfo>* truths) {
	cv::Mat temp = cv::Mat::zeros(1024, 1024, CV_8UC3);

	resize(orig_img, temp, temp.size());
	int w = temp.cols;
	int h = temp.rows;

	int step_w = w / stride;
	int step_h = h / stride;

	for (int y = 0; y < h; y += step_h) {
		cv::Point2i start(0, y), stop(w - 1, y);
		cv::line(temp, start, stop, color);
	}
	for (int x = 0; x < w; x += step_w) {
		cv::Point2i start(x, 0), stop(x, h - 1);
		cv::line(temp, start, stop, color);
	} 
	for (int i = 0; i < (int)truths->size(); i++) {
		const ObjectInfo& truth = truths->at(i);
		int x = (int)(truth.x * stride);
		int y = (int)(truth.y * stride);
		int p_x = (int)(truth.x * w);
		int p_y = (int)(truth.y * h);
		cv::Point2i pos( p_x - 30, p_y - 10);
		char text[100];
		sprintf(text, "(%d,%d)", x, y);
		string str(text);
		cv::Scalar text_color(23, 35, 60);
		cv::putText(temp, str, pos, cv::FONT_HERSHEY_COMPLEX, 0.8, text_color, 2);
		cv::circle(temp, cv::Point2i(p_x, p_y), 5, cv::Scalar(0, 0, 255), 5);
		
	}
	vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);
	params.push_back(90);
	char filename[MAX_PATH];
	sprintf(filename, "truth_in_%02d.jpg", stride);
	cv::imwrite(filename, temp, params);


}
bool CNNNetwork::Detect(const char* filename) {
	if (input_channels <= 0 || input_width <= 0 || input_height <= 0 || !filename)
		return false;
	
#if 0 
	string s(filename);
	ifstream f(replace_extension(s, ".txt"));
	if (!f.is_open()) {
		cerr << " Error: label image `" << s << "` does not exist!\n";
		return false;
	} 

	int i = 0;
	int x, y;
	int i_13, j_13, mod_i_13, mod_j_13;
	int i_26, j_26, mod_i_26, mod_j_26;
 
	ObjectInfo truth;
	truths[0]->clear();
	while (getline(f, s)) { 
		const char* ptr = s.c_str();
		
		truth.class_id = get_next_int(ptr);
		truth.x = get_next_float(ptr);
		truth.y = get_next_float(ptr);
		truth.w = get_next_float(ptr);
		truth.h = get_next_float(ptr);
 
		x = (int)(truth.x * input_width);
		y = (int)(truth.y * input_height);
		i_13 = x / 32; mod_i_13 = x % 32; j_13 = y / 32; mod_j_13 = y % 32;
		i_26 = x / 16; mod_i_26 = x % 16; j_26 = y / 16; mod_j_26 = y % 16;

		cout << " The " << i << "th truth data:(" << x << ", " << y << 
			"), object in [" << i_13 << "," << j_13 <<
			  "] / 13x13,[" << i_26 << "," << j_26 << "] / 26x26.\n";
		i++; 

		truths[0]->push_back(truth);
	}
	f.close(); 
#endif
	cv::Mat mat = cv::imread(filename);

	
	//DrawTruthInGrid(mat, 32, cv::Scalar(40, 20, 230), truths[0]);
	//DrawTruthInGrid(mat, 16, cv::Scalar(17, 83, 230), truths[0]);


	if (mat.empty()) {
		cerr << "\nError: Cannot load image `" << filename << "`!\n";
		return false;
	}
	Image image(reinterpret_cast<uint8_t*>(mat.data), mat.cols, mat.rows, mat.channels());
	 
	if (!image.ResizeTo(input_width, input_height) || 
		!image.PullFromGPU()) {
		cerr << " Error: failed to resize image to "<< input_width << " x " << 
			input_height << " detection task!\n";
		return false;
	} 
	if (input) delete[]input;
	input = image.GetData();
	input_channels = image.GetChannels(); 

	ForwardContext context = { false, false, false, false, nullptr};
	detections.clear();
	for (int i = 0; i < (int)layers.size(); i++) {
		Layer* l = layers[i];
 
		if(!l->FuseBatchNormModule()){
			cerr << " Error: FuseBatchNorm failed in  " << l->GetName() << "!\n"; 
			input = nullptr;
			return false;
		} 
		if (!l->Forward(context)) {
			cerr << " Error: Forward failed in  " << l->GetName() << "!\n"; 
			input = nullptr;
			return false;
		}
	}
	input = nullptr;
	string name;
	int l, r, t, b;
	cv::Scalar color(30, 70, 220 );
	for (int i = 0; i < (int)detections.size(); i++) {
		DetectionResult& dr = detections[i];
		GetAppConfig().GetClass(dr.class_id, name);
		l = (int)((dr.x - dr.w * 0.5f) * mat.cols);
		r = (int)((dr.x + dr.w * 0.5f) * mat.cols);
		t = (int)((dr.y - dr.h * 0.5f) * mat.rows);
		b = (int)((dr.y + dr.h * 0.5f) * mat.rows);
		cv::Point ptTopLeft(l, t),  ptBottomRight(r, b);
		cv::rectangle(mat, ptTopLeft, ptBottomRight, color, 2); 
		
		char conf_text[20];
		if (dr.confidence == 1.0f)
			strcpy(conf_text, ": 100%");
		else {
			sprintf(conf_text, ": %.2f%%", dr.confidence * 100.0f);
		}
		ptTopLeft.y -= 15;
		cv::putText(mat, name + conf_text, ptTopLeft, cv::FONT_HERSHEY_COMPLEX, 0.8, color, 2);
		int baseline = 0;
		cv::Size size =  cv::getTextSize(name + conf_text, cv::FONT_HERSHEY_COMPLEX, 0.8, 2, &baseline);
		cout << name << " at [" << l << ", " << r << ", " << t << ", " << b << " ] "  << conf_text << endl;
	}
	vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);
	params.push_back(90);
	
	cv::imwrite("predictions.jpg", mat, params);

	cout << " INFO: output written to predictions.jpg!\n";
	
	return true;
}