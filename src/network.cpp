#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"
#include "param_pool.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image.h"
#include "box.h" 
#include <direct.h>
#include "yolo.h"
#include "OpenVINO.h"

 
CNNNetwork::CNNNetwork() {
	data_type = CUDNN_DATA_FLOAT; 
	data_format = CUDNN_TENSOR_NCHW; 
	def_actvation = "leaky";
	truths = nullptr;
	input = nullptr;
	cur_iteration = 0;
	detection_layers = 2;  
	nms_type = DEFAULT_NMS;
	current_training_rotates = nullptr; 
	 
}

CNNNetwork::~CNNNetwork() {
	for (auto l : layers) {
		delete l;
	}
//	if (workspace) cudaFree(workspace);
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
	if (current_training_rotates) delete[]current_training_rotates;
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
	//float gt_iou = BoxIoU(box, gt_box);
	while (index >= 0) {
		DetectionResult &dr = detections[index];
		Box box2(dr.x, dr.y, dr.w, dr.h);
		float iou = BoxIoU(box, box2);
		if ( iou > GetAppConfig().NMSThreshold() ) {
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
	
	const char* temp = root->Attribute("name");
	if (temp) {		
		name = temp;
	}
	else {
		name = file_part(filename);
		replace_extension(name, "");
	}
	temp = root->Attribute("nms-type");
	if (temp) {
		if (0 == _strcmpi("greedy", temp))
			nms_type = GREEDY_NMS;
		else if(0 == _strcmpi("diou", temp))
			nms_type = DIOU_NMS;
		else if(0 == _strcmpi("corners", temp))
			nms_type = CORNERS_NMS;
	}
	temp = root->Attribute("default-activation");
	if (temp) {
		def_actvation = temp;
	}
	 
	struct stat s = { 0 };
	stat(name.c_str(), &s);
	if (0 == (s.st_mode & S_IFDIR)) {
		_mkdir(name.c_str());
	}
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
	XMLElement* classElement = root->FirstChildElement("classes/class");
	while (classElement) {
		const char* class_name = classElement->GetText();
		if (!class_name) class_name = "<NONAME>";
		classes.push_back(class_name);
		classElement = classElement->NextSiblingElement();
	}
	if (classes.size() == 0) {
		cerr << " Error: class count should > 0 !\n";
		return false;
	}
	XMLElement* anchorElement = root->FirstChildElement("anchors");
	const char* anchor_style = anchorElement->Attribute("style");
	if (!anchor_style || strcmp(anchor_style, "ssd")) {
		anchorElement = anchorElement->FirstChildElement("anchor");
		if (anchorElement) {
			while (anchorElement) {
				float w = anchorElement->FloatAttribute("width", 0.0f) ;
				float h = anchorElement->FloatAttribute("height", 0.0f) ;
				if (w > 0.0 && h > 0.0) {
					anchors.push_back(pair<float, float>(w, h));
				}
				anchorElement = anchorElement->NextSiblingElement();
			}
		}
	}
	else {	

			float s_min = 0.2f, s_max = 0.9f;
			int scales_count = 6;
			bool add_default_scale = true;
			vector<float> aspect_ratios;
			anchorElement->QueryFloatText("scale-min",s_min);
			anchorElement->QueryFloatText("scale-max", s_max);
			anchorElement->QueryIntText("detection-layers", detection_layers);
			anchorElement->QueryIntText("default-box-scales", scales_count);
			switch (scales_count) {
			case 6:
				aspect_ratios = { 1.0f, 2.0f, 3.0f, 1.0f/3.0f, 0.5f };
				break;
			case 4:
				aspect_ratios = { 1.0f, 2.0f, 0.5f };
				break;
			case 5:
				add_default_scale = false;
				aspect_ratios = { 1.0f, 2.0f, 3.0f, 1.0f / 3.0f, 0.5f };
				break;
			case 3:
				add_default_scale = false;
				aspect_ratios = { 1.0f, 2.0f, 0.5f };
			default:
				cerr << " Error: default-box-scales must be one in 3,4,5,6!\n";
				return false;
			}
			if (detection_layers <= 1) {
				cerr << " Error: detection-layers must bigger than 1!\n";
				return false;
			}
			vector<float> sks;
			for (int l = 0; l <= detection_layers; l++) {
				sks.push_back(s_min + l *(s_max - s_min) / (detection_layers - 1));
			}
			float w, h,t;
			for (int i = 0; i < detection_layers; i++) {
				if (add_default_scale) {
					w = sqrtf(sks[i] * sks[i + 1]);
					anchors.push_back(pair<float, float>(w, w));
				}
				for (int j = 0; j < (int)aspect_ratios.size(); j++) {
					t = sqrtf(aspect_ratios[j]);
					w = sks[i] * t;
					h = sks[i] / t;
					anchors.push_back(pair<float, float>(w, h));
				}
			} 

	}
	XMLElement* layerElement = root->FirstChildElement("layers/layer"); 
	InferenceModule* last_module = nullptr;
	while (layerElement) {
		Layer* layer = NULL;
		try {
			layer = New Layer(layerElement,  this, last_module);
			layers.push_back(layer);
			layerElement = layerElement->NextSiblingElement();
		}
		catch (const char* t) {
			cerr << "Unrecognized layer type `" << t << "`! \n";
			return false;
		}
		
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

bool CNNNetwork::GetAnchor(int index, float & width, float & height, bool normalized) {
	if (index < 0 || index >(int)anchors.size()) return false;
	if (normalized) {
		width = anchors[index].first / input_width;
		height = anchors[index].second / input_height;
	}
	else {
		width = anchors[index].first;
		height = anchors[index].second;
	}
	return true;
}

bool CNNNetwork::Forward(bool training) {
	ForwardContext context = { training,
		GetAppConfig().ConvParamsFreezed(),
		GetAppConfig().BNParamsFreezed(),
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

bool CNNNetwork::Train(bool restart ) { 
	DataLoader loader;

	string filename("loss_");
	filename += get_time_str();
	filename += ".log"; 
	 
	if (restart)
		cur_iteration = 0;
	else
		cur_iteration =  weights_pool.GetIteration();

	int total_images = cur_iteration * GetAppConfig().GetBatch();

	cout << "\n *** Start training from iteration " << (cur_iteration + 1) << "...\n\n";
	int new_width, new_height;
	char weights_file[MAX_PATH];
	float avg_loss = -1.0;   
	
	int input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
	if(!current_training_rotates) current_training_rotates = New RotateType[mini_batch + 1]; 
	while (!GetAppConfig().IsLastIteration(cur_iteration)) { 
		cur_iteration++;
		clock_t start_clk = clock(); 
		int total_objects = 0; 
		training_results.clear();
		cout << " -- Interation " << cur_iteration << endl;
		for (int i = 0; i < GetAppConfig().GetSubdivision(); i++, total_images += mini_batch) {
			
			current_training_files.clear(); 
			long t = GetTickCount(); 
			memset(input, 0,  input_len);
			if (!loader.MiniBatchLoad(input, truths, input_channels, mini_batch, input_width, input_height, (int)classes.size(), 
				&current_training_files, current_training_rotates)) {
				return false;
			}  
			for (int i = 0; i < mini_batch; i++) {
				total_objects += truths[i]->size();
			}
			
			if (!Forward(true)) return false;  
			if (!Backward()) return false;   
		}
		float loss = 0.0f, box_loss = 0.0f, obj_loss = 0.0f, cls_loss = 0.0f, iou = 0.0f;
		stringstream ss; 

		for (TrainingResult& r : training_results) {
			loss += (r.obj_loss + r.box_loss + r.cls_loss); 
			box_loss += r.box_loss;
			obj_loss += r.obj_loss;
			cls_loss += r.cls_loss;
			iou += r.iou;
		}
		loss /= training_results.size(); 
		box_loss /= training_results.size();
		obj_loss /= training_results.size();
		cls_loss /= training_results.size();
		iou /= total_objects;

		if (avg_loss < 0)
			avg_loss = loss;
		else
			avg_loss = avg_loss * 0.9 + loss * 0.1;
		int ms = (clock() - start_clk) * 1000 / CLOCKS_PER_SEC;		 
		float lr = GetAppConfig().GetCurrentLearningRate(cur_iteration);
		cout << "  * loss : " << fixed << setprecision(4) << loss << " [" << obj_loss << "(o)+" << box_loss << "(b)";
		if (classes.size() > 1)
			cout << "+" << cls_loss << "(c)";
		cout << "], avg-loss: " << avg_loss << ", iou: " << iou << ", lr:" << defaultfloat << lr << ", " << ms << "ms, " << total_images << " images.\n\n";
		
		
		ofstream ofs(filename,ios::app);
		if (ofs.is_open()) {
			//sprintf(info, "")
			ofs << cur_iteration << ",\t" << input_width  << ",\t" << input_height  << ",\t" <<  lr << ",\t"
				<< fixed << setprecision(4) << loss << ",\t"  << avg_loss <<  endl;
			ofs.close();
		}
		
		for (auto l : layers) {
			if (!l->Update(lr)) return false;
		} 
		if (GetAppConfig().SaveIteration(cur_iteration)) {
			sprintf_s(weights_file, MAX_PATH, "%s/%s-%d.rweights", name.c_str(), def_actvation.c_str(), cur_iteration);
			if (weights_pool.Save(weights_file, cur_iteration)) {
 				cout << " INFO: Save weights to `" << weights_file << "` ! \n";
			}
			if (GetAppConfig().UpdatePolicy() == Adam) {
				sprintf_s(weights_file, MAX_PATH, "%s/%s-%d.adam.rweights", name.c_str(), def_actvation.c_str(), cur_iteration);
				if (!adam_weights_pool.Save(weights_file, cur_iteration)) {
					cerr << " Warning: Save adam weights to `" << weights_file << " failed` ! \n";
				}
			}
			
		} 
		if (GetAppConfig().RadmonScale(cur_iteration, new_width, new_height) &&
			(new_width != input_width || new_height != input_width) ) {
			cout << " *** Input Resizing to " << new_width << "x" << new_height << " ...\n";
			input_width = new_width;
			input_height = new_height;
			input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
			input = (float*)realloc(input, input_len);

		} 
		
	}
	//TODO : save final 
	cout << "\n INFO: Training done.\n";
	return true;
}

bool CNNNetwork::CreateOpenVINOIRv7(const string& dir, const string& ir_name, bool fp16) {
	 
	string prefix = dir;
	if (prefix.find_last_of('\\') != prefix.length() - 1 &&
		prefix.find_last_of('/') != prefix.length() - 1)
		prefix += SPLIT_CHAR;

	struct stat s = { 0 };
	stat(prefix.c_str(), &s);
	if (!(s.st_mode & _S_IFDIR)) {
		if (!_mkdir(prefix.c_str())) {
			cerr << "\n Error: Failed to create directory `" << prefix.c_str() << "`!\n\n";
			return false;
		}
	}
	prefix += ir_name;
	vector<InferenceModule*> modules;
	InferenceModule* last_mod = nullptr;
	 
	for (auto l : layers) {
		if (!l->FuseBatchNormModule()) {
			cerr << " Error: FuseBatchNorm failed in  " << l->GetName() << "!\n";
			return false;
		}
		for (auto m : l->modules) {
			if (!dynamic_cast<BatchNormModule*>(m)) {
				modules.push_back(m);
				
			}
		}
	}
	vector<OpenVINOIRv7Layer> ir_layers;
	vector<OpenVINOIRv7Edge> ir_edges;

	OpenVINOIRv7Layer inputs(0, "inputs", "Input", "FP32") ;	 
	inputs.outputs.push_back({ 0,1,input_channels, input_height, input_width });
	ir_layers.push_back(inputs);

	ofstream  xml(prefix + ".xml", ios::trunc);
	ofstream  bin(prefix + ".bin", ios::binary | ios::trunc);
	if ((!xml.is_open()) || (!bin.is_open())) return false;
	size_t offset = 0;
	for (auto m : modules) {
		if (!m->RenderOpenVINOIR(ir_layers, ir_edges, bin, offset,fp16)) {
			cerr << "\n Error: Render IR failed for " << m->Name() << "!\n";
			return false;
		}
	}
	bin.close();
 
	//TODO: check duplicated split modules
	int index =  ir_layers.size() - 1;
	while(index > 0) {
		OpenVINOIRv7Layer& l_p = ir_layers[index - 1];
		OpenVINOIRv7Layer& l = ir_layers[index];

		if (l.ltype == "Split" && l_p.ltype == "Split") {
			// remove l
			int remove_id = l.id; 
			int replace_id = l_p.id;
			ir_layers.erase(ir_layers.begin() + index);
			for (auto& e : ir_edges) {
				if (e.from_layer == remove_id) e.from_layer = replace_id;
				if (e.to_layer == remove_id) e.to_layer = -1;
			}
		} 
		index--;
	} 
	xml << "<?xml version=\"1.0\" ?>\n";
	xml << "<net batch = \"1\" name=\"" << ir_name << "\" version=\"7\">\n\t<layers>\n";

	for (OpenVINOIRv7Layer& l : ir_layers) {
		xml << l;
	}

	xml << "\t</layers>\n\t<edges>\n";

	for (OpenVINOIRv7Edge& e : ir_edges) {
		if (e.to_layer > 0) {
			xml << e;
		}
	}
	xml << "\t</edges>\n\t<meta_data>\n\t</meta_data>\n</net>";
	xml.close();

	xml.open(prefix + ".yolo.xml", ios::trunc);
	xml << "<?xml version=\"1.0\" ?>\n";
	xml << "<net version=\"1.0\" >\n\t<classes>\n"; 
	for (auto& c : classes) {
		xml << "\t\t<class>" << c << "</class>\n";
	}
	xml << "\t</classes>\n\t<anchors>\n";
	for (auto& a : anchors) {
		xml << "\t\t<anchor width=\"" << a.first << "\" height=\"" << a.second << "\" />\n";
	}
	xml << "\t</anchors>\n\t<outputs>\n";
	for (auto m : modules) {
		m->WriteOpenVINOOutput(xml);
	}
	xml << "\t</outputs>\n</net>\n";
	xml.close();
	return true;
} 
void CNNNetwork::GetAnchorsStr(string & str) const {
	str = "";
	for (int i = 0; i < (int)anchors.size(); i++) {
		auto& a = anchors[i]; 
		str += to_string((int)a.first) + "," + to_string((int)a.second);
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

static uchar calc_color(double temp1, double temp2, double temp3) {

	double color;
	if (6.0 * temp3 < 1.0)
		color = temp1 + (temp2 - temp1) * 6.0* temp3;
	else if (2.0 * temp3 < 1.0)
		color = temp2;
	else if (3.0 * temp3 < 2.0)
		color = temp1 + (temp2 - temp1) * ((2.0 / 3.0) - temp3) * 6.0;
	else
		color = temp1;

	return (uchar)(color * 255);
}
void hsl2rgb(double hue, double sat, double light, uchar* rgb) {
	if (0.0 == sat) {
		rgb[0] = (uchar)(light * 255);
		rgb[1] = rgb[0];
		rgb[2] = rgb[0];
		return;
	}
	double temp1, temp2, temp3;
	if (light < 0.5) {
		temp2 = light * (1.0 + sat);
	}
	else {
		temp2 = light + sat - light * sat;
	}
	temp1 = 2.0 * light - temp2;
	temp3 = hue + 1.0 / 3.0;//for R, temp3=H+1.0/3.0	
	if (temp3 > 1.0)
		temp3 = temp3 - 1.0;
	rgb[0] = calc_color(temp1, temp2, temp3);
	temp3 = hue; //for G, temp3=H
	rgb[1] = calc_color(temp1, temp2, temp3);
	temp3 = hue - 1.0 / 3.0;//for B, temp3=H-1.0/3.0
	if (temp3 < 0.0)
		temp3 = temp3 + 1.0;
	rgb[2] = calc_color(temp1, temp2, temp3);

}

bool CNNNetwork::Detect(const char* path) {
	if (input_channels <= 0 || input_width <= 0 || input_height <= 0 || !path)
		return false;

	vector<string> files;
	struct stat s = { 0 }; 
	string folder(path);
	stat(path, &s);
	if (s.st_mode & _S_IFDIR) {		
		char c = path[folder.length() - 1];
		if (c != '/' && c != '\\')
			folder += SPLIT_CHAR; 
		string search_str = folder + "*.*";
		_finddata_t find_data;
		intptr_t handle = _findfirst(search_str.c_str(), &find_data);
		if (handle == -1) {
			cerr << "Error: Failed to find first file under `" << folder.c_str() << "`!\n";
			return false;
		}
		bool cont = true;

		while (cont) {
			if (0 == (find_data.attrib & _A_SUBDIR)) {
				if (is_suffix(find_data.name, ".jpg") ||
					is_suffix(find_data.name, ".JPG") ||
					is_suffix(find_data.name, ".png") ||
					is_suffix(find_data.name, ".PNG") ||
					is_suffix(find_data.name, ".bmp") ||
					is_suffix(find_data.name, ".BMP")
					) {
					files.push_back(folder + find_data.name);
				}
			}
			cont = (_findnext(handle, &find_data) == 0);
		}
		_findclose(handle);

	}
	else
		files.push_back(path);
	char output_path[MAX_PATH];
	sprintf_s(output_path, MAX_PATH, "predictions@%.2f", GetAppConfig().ThreshHold());
	stat(output_path, &s);
	if (0 == (s.st_mode & _S_IFDIR)) {
		_mkdir(output_path);
	}
	for (auto l : layers) {
		if (!l->FuseBatchNormModule()) {
			cerr << " FuseBatch error @ " << l->GetName() << "! \n\n";
			return false;
		}
	}
	for (auto it : files) {
		const char* filename = it.c_str();
		std::cout << " Processing " << filename << " ...\n";
		cv::Mat mat = cv::imread(filename);
		if (mat.empty()) {
			cerr << "\nError: Cannot load image `" << filename << "`!\n";
			return false;
		}
		if (input) delete[]input;

		cv::Size sz(input_width, input_height);
		cv::Mat temp = cv::Mat::zeros(sz, CV_8UC3);
		resize(mat, temp, sz);

		input_channels = temp.channels();
		CpuPtr<float> buffer(input_channels * input_width * input_height);
		input = buffer.ptr;
		int index = 0;
		for (size_t c = 0; c < input_channels; c++) {
			for (size_t h = 0; h < input_height; h++) {
				for (size_t w = 0; w < input_width; w++) {
					input[index++] = (float)temp.at<cv::Vec3b>(h, w)[c] / 255.0f;
				}
			}
		}

		ForwardContext context = { false, false, false, nullptr };
		detections.clear();
		for (auto l : layers) { 
			if (!l->Forward(context)) {
				cerr << " Error: Forward failed in  " << l->GetName() << "!\n";
				input = nullptr;
				return false;
			}
		}
		input = nullptr;

		int l, r, t, b;
		cv::Mat overlay = cv::Mat::zeros(mat.size(), CV_8UC3);
		
		uchar rgb[3];
		vector<cv::Scalar> colors;
		for (int i = 0; i < (int)detections.size(); i++) { 
			hsl2rgb(0.6 + (i + 1.0f) / detections.size() * 0.4 , 1.0, 0.5, rgb);
			cv::Scalar color(rgb[0], rgb[1], rgb[2]);
			colors.push_back(color);
		}
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(colors.begin(), colors.end(), default_random_engine(seed));

		for (int i = 0; i < (int)detections.size(); i++) {
			DetectionResult& dr = detections[i];
			const string& name = classes[dr.class_id];
			l = (int)((dr.x - dr.w * 0.5f) * mat.cols);
			r = (int)((dr.x + dr.w * 0.5f) * mat.cols);
			t = (int)((dr.y - dr.h * 0.5f) * mat.rows);
			b = (int)((dr.y + dr.h * 0.5f) * mat.rows);
			cv::Point ptTopLeft(l, t), ptBottomRight(r, b);
			
			
			cv::rectangle(mat, ptTopLeft, ptBottomRight, colors.at(i), 1);
			cv::rectangle(overlay, ptTopLeft, ptBottomRight, colors.at(i), -1);
			
			char conf_text[20];
			if (dr.confidence == 1.0f)
				strcpy(conf_text, "100%");
			else {
				sprintf(conf_text, "%.2f%%", dr.confidence * 100.0f);
			}
			int baseline = 0;
			cv::Size size = cv::getTextSize( conf_text, cv::QT_FONT_NORMAL, 1.0, 2, &baseline);
			ptTopLeft.y = ((t + b) >> 1) + (size.height >> 1);
			ptTopLeft.x = ((l + r) >> 1) - (size.width >> 1);
			cv::putText(mat, conf_text, ptTopLeft, cv::QT_FONT_NORMAL, 1.0, colors.at(i), 1);
			//
			//
			cout << name << " at [" << l << ", " << r << ", " << t << ", " << b << " ] " << conf_text << endl;
		} 
		vector<int> params;
		params.push_back(cv::IMWRITE_JPEG_QUALITY);
		params.push_back(90);
		cv::addWeighted(mat, 1.0, overlay, 0.3, 0, mat);
		sprintf_s(output_path, MAX_PATH, "predictions@%.2f/%s", GetAppConfig().ThreshHold(), file_part(it));
 
		cv::imwrite(output_path, mat, params);
	}

	std::cout << " INFO: output written to predictions/!\n";
	
	return true;
}
static bool load_truths(const char* filename, vector<ObjectInfo>& gts,int classes) {
	gts.clear();
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Can't open label file. (This can be normal only if you use MSCOCO) \n";
		//file_error(filename);
		ofstream badfile("bad.list", ios::app);
		if (badfile.is_open()) {
			string str(filename);
			str += "\n";
			badfile.write(str.c_str(), str.length());
			badfile.close();
		}
		return false;
	}
	char line[512];
	int i = 0;
	ObjectInfo oi;
	while (!file.eof()) {
		i++;
		file.getline(line, 512);
		if (0 == *line) continue;
		const char* line_str = line;
		oi.class_id = get_next_int(line_str);
		if (oi.class_id >= classes) {
			cerr << "abnormal line data in " << filename << "(" << i << ")! class_id is " << oi.class_id << "!" << endl;
			continue;
		}
		oi.x = get_next_float(line_str);
		oi.y = get_next_float(line_str);
		oi.w = get_next_float(line_str);
		if (0 == *line_str) {
			cerr << "abnormal line data in " << filename << "(" << i << ")!" << endl;
			continue;
		}
		oi.h = get_next_float(line_str);
		gts.push_back(oi);
	}
	file.close();
	return true;
}
//enum PredResultType {TP,FP,TN,FN};
struct PredictionInfo {
	float confidence;
	float best_iou;
	int class_id;
	int file_index;
	int gt_index;
	bool true_positive;
	//PredResultType result;
};
bool compare_confidence(PredictionInfo p1, PredictionInfo p2) {
	return p1.confidence > p2.confidence;
}
bool CNNNetwork::Eval() {
	if (GetAppConfig().GetDatasetCount() == 0) {
		cerr << " *** No dataset definitions in the configuration! *** \r\n";
		return false;
	}
	if (input) delete[]input;
	
	for (int i = 0; i < GetAppConfig().GetDatasetCount(); i++) {
		int total_ground_truths = 0;
		vector<PredictionInfo> performance_data;
		const Dataset* ds = GetAppConfig().GetDataSet(i);
		std::cout << "  Processing dataset `" << ds->GetName() << "`...\n";
		Image img;
		for (size_t j = 0; j < ds->GetSize(); j++) {
			string filename = ds->FilenameAt(j);
			std::cout << "   Loading `" << filename << "`...\n";
			if (!img.Load(filename.c_str(), input_channels)) {
				cerr << "   Load failed!\n\n";
				continue;
			}
			if (!img.ResizeTo(input_width, input_height, true)) {
				cerr << "   Resize failed!\n\n";
				continue;
			}
			vector<ObjectInfo> gts;
			if (!load_truths(replace_extension(filename, ".txt"), gts, classes.size()))
				continue;
			total_ground_truths += gts.size();
			img.PullFromGPU();
			detections.clear();
			
			input = img.GetData();
			if (!Forward(false)) {
				input = nullptr;
				return false;
			}
			PredictionInfo pdi;
			unsigned int prev_predicts = performance_data.size();
			for (int d = 0; d < detections.size(); d++) {
				auto det = detections.at(d);				 
				Box det_box(det.x, det.y, det.w, det.h);
				pdi.best_iou = 0.0f;
				pdi.gt_index  = -1;
				pdi.class_id = -1;
				for (int g = 0; g < gts.size(); g++) {
					ObjectInfo& gt = gts[g];
					Box gt_box(gt.x, gt.y, gt.w, gt.h);
					float iou = BoxIoU(gt_box, det_box);
					if (iou > pdi.best_iou) {
						pdi.best_iou = iou;
						pdi.gt_index = g; 
					}
				}
				pdi.confidence = det.confidence;
				pdi.class_id = det.class_id;
				pdi.file_index = j;
				pdi.true_positive = false;
				//什么样的情况是true_positive?下面来整理一下
				if (pdi.gt_index != -1) {
					if (det.class_id == gts[pdi.gt_index].class_id) {
						pdi.true_positive = pdi.best_iou > GetAppConfig().ThreshHold(); 
					}
					else
						pdi.class_id = -1;
				} 
				for (unsigned int temp = prev_predicts ; temp < performance_data.size(); temp++) {
					PredictionInfo& prev_p = performance_data[temp];
					if (prev_p.gt_index == pdi.gt_index ) {
						//对同一个gt预测的，只有iou 最大的才是tp
						if (prev_p.best_iou > pdi.best_iou) {
							pdi.true_positive = false;
						}
					}
				}
				performance_data.push_back(pdi);
			}

			input = nullptr;
		}
		// 按confidence从大到小排序
		sort(performance_data.begin(), performance_data.end(), compare_confidence);
		
		// 用11点法计算mAP
		float mAP = 0.0f;
		for (int c = 0; c < classes.size(); c++) {
			vector< pair<float, float> >prs;
			float acc_TP = 0.0f, acc_FP = 0.0f;
			float max_p = 0.0f, cur_base = 0.0f, best_p = 0.0f, avr_iou = 0.0f;
			float r;
			for (auto pdi : performance_data) {
				if(pdi.class_id != c ) continue;
				if (pdi.true_positive) {
					acc_TP += 1.0f;
					avr_iou += pdi.best_iou;
				}
				else
					acc_FP += 1.0f;
				float p = acc_TP / (acc_TP + acc_FP);
				r = acc_TP / total_ground_truths;				
				if (r - cur_base > 0.1f) {
					prs.push_back(make_pair(cur_base,best_p ));
					cur_base += 0.1f;
					best_p = 0.0f;
				}
				else 
					if (p > best_p) best_p = p;
			}
			//if (r != cur_base) {
			prs.push_back(make_pair(cur_base, best_p));
			//}
			cout << "AP for " << classes[c].c_str() << " @ Threshold "<< fixed << setprecision(2) 
				<< GetAppConfig().ThreshHold() <<": \n  ";
			float ap = 0.0f;
			for (auto pr : prs) {
				cout << setprecision(2) << pr.first << "(" << setprecision(6) << pr.second << ") ";
				ap += pr.second;
			}
			ap /= 0.11f;
			avr_iou /= (acc_TP > 0)? acc_TP : 1.0f;
			cout << "\n - " << fixed << setprecision(2) << ap  << "%, average iou:" << 
				setprecision(4) << avr_iou<< endl;
			mAP += ap;
		}
		mAP /= classes.size();
		cout << "\n *** mAP is " << setprecision(2) <<  mAP << "%"<<  endl;
	}
	return true;
}
bool CNNNetwork::Eval_old() { 
	
	if (GetAppConfig().GetDatasetCount() == 0) {
		cerr << " *** No dataset definitions in the configuration! *** \r\n";
		return false;
	} 
	if (input) delete[]input;
	vector<ObjectInfo> gts;
	string filename(name);
	filename += ".test.results.csv";
	ofstream of(filename.c_str(),ios::trunc);
	of << "Threshold, Avg IoU, Avg Conf, R@.5IoU, P@.5IoU, R@.75IoU, P@0.75IoU\n";
	for (int i = 0; i < GetAppConfig().GetDatasetCount(); i++) { 
		const Dataset* ds = GetAppConfig().GetDataSet(i);
		int total_ground_truths = 0;
		int total_recall50[10],total_recall75[10],total_predictions[10];
		float total_iou[10],total_conf[10];
		memset(total_recall50, 0, sizeof(int) * 10);
		memset(total_recall75, 0, sizeof(int) * 10);
		memset(total_predictions, 0, sizeof(int) * 10);
		memset(total_iou, 0, sizeof(float) * 10);
		memset(total_conf, 0, sizeof(float) * 10);

		std::cout << "  Processing dataset `" << ds->GetName() << "`...\n";
		Image img;

		
		of << ds->GetName().c_str() << ",,,,,,\n" << fixed << setprecision(4);
		
		for (size_t j = 0; j < ds->GetSize(); j++) {
			string filename = ds->FilenameAt(j);
			std::cout << "   Loading `" << filename << "`...\n";
			if (!img.Load(filename.c_str(), input_channels)) {
				cerr << "   Load failed!\n\n";
				continue;
			}
			if (!img.ResizeTo(input_width, input_height, true)) {
				cerr << "   Resize failed!\n\n";
				continue;
			}
			if (!load_truths(replace_extension(filename, ".txt"), gts, classes.size()))
				continue;
			total_ground_truths += gts.size();
			img.PullFromGPU();
			
			int index = 0;
			for (float th = 0.5; th < 1.0f; th += 0.05f,index ++) {
				detections.clear();
				input = img.GetData();
				GetAppConfig().ThreshHold(th);
				if (!Forward(false)) return false;
				input = nullptr;
				int recall_50 = 0, recall_75 = 0, t_i = 0;
				total_predictions[index] += detections.size();
				CpuPtr<float> ious(detections.size());
				CpuPtr<int> hits(gts.size());
				for (auto it = gts.begin(); it != gts.end(); it++, t_i++) {
					Box gt_box(it->x, it->y, it->w, it->h);
					float best_iou = 0.0f;
					int best_index = -1;
					for (int d = 0; d < detections.size(); d++) {
						auto det = detections.at(d);
						if (det.class_id != (int)it->class_id) continue;
						Box det_box(det.x, det.y, det.w, det.h);
						float iou = BoxIoU(gt_box, det_box);
						if (iou > best_iou) {
							best_iou = iou;
							best_index = d;
						}
					}
					if (-1 == best_index) continue;// not recalled
					if (ious.ptr[best_index] != 0.0f) {
						// 1 detection refers to more than 1 gt
						if (ious.ptr[best_index] > best_iou) continue; // previous gt has better iou
						for (int t = 0; t < hits.Length(); t++) {
							if (hits.ptr[t] + 1 == best_index)
								hits.ptr[t] = 0;
						}
						ious.ptr[best_index] = best_iou;
						hits.ptr[t_i] = best_index + 1;
					}
					else
						ious.ptr[best_index] = best_iou;
				}
				for (int n = 0; n < ious.Length(); n++) {
					if (ious.ptr[n] >= 0.75f) {
						recall_75++;
					}
					if (ious.ptr[n] >= 0.5f) {
						recall_50++;
						total_iou[index] += ious.ptr[n];
						total_conf[index] += detections.at(n).confidence;
					}
				}
				total_recall50[index] += recall_50;
				total_recall75[index] += recall_75;
			}
		}
		int index = 0;
		for (float th = 0.5f; th < 1.0f; th += 0.05f, index++) {
			float r50 = total_recall50[index] * 100.0f / total_ground_truths;
			float r75 = total_recall75[index] * 100.0f / total_ground_truths;
			float p50 = total_recall50[index] * 100.0f / total_predictions[index];
			float p75 = total_recall75[index] * 100.0f / total_predictions[index];
			float avg_iou = total_iou[index] / total_ground_truths;
			float avg_conf = total_conf[index] / total_ground_truths;
			of << th << ", " << avg_iou << ", " << avg_conf << ", " << r50 << ", " << p50 << ", " << r75 << ", " << p75 << "\n";
		} 
	}
	of.close();
	return true;
}
