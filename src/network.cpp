#include <array>
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

 
CNNNetwork::CNNNetwork(int h ,int w) {
	data_type = CUDNN_DATA_FLOAT; 
	data_format = CUDNN_TENSOR_NCHW; 
	def_actvation = "leaky";
	truths = nullptr;
	input = nullptr;
	cur_iteration = 0;
	detection_layers = 2;  
	nms_type = DEFAULT_NMS;
	current_training_rotates = nullptr; 
	override_height = h;
	override_width = w;
	 
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
			cout << "\n Data type: FP16. ";
		else
			cout << "\n Data type: FP32. ";
		input_channels = 3;
		input_width = 416;
		input_height = 416;
		inputElement->QueryIntText("channels", input_channels);
		if (override_height > 0)
			input_height = override_height;
		else
			inputElement->QueryIntText("height", input_height);

		if (override_width > 0)
			input_width = override_width;
		else 
			inputElement->QueryIntText("width", input_width);
		int elements = mini_batch * input_channels * input_width * input_height;
		input = New float[elements];
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
	float bflops = 0.0f;
	while (layerElement) {
		Layer* layer = NULL;
		try {
			layer = New Layer(layerElement,  this, last_module);
			for (auto& m : layer->modules) {
				bflops += m->GetFlops();
			}
			layers.push_back(layer);
			layerElement = layerElement->NextSiblingElement();
		}
		catch (const char* t) {
			cerr << "Unrecognized layer type `" << t << "`! \n";
			return false;
		}
		
	}
	bflops /= 1000000000.0f;
	cout << " Input dimensions: [" << mini_batch << ", " << input_channels << ", " << input_width << ", " << input_height
		<< "], Computational complexity: " << fixed << setprecision(3) << bflops << " billion flops. \n";

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
			if (GetAppConfig().UpdatePolicy() == Adam && GetAppConfig().SaveAdamParams()) {
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
	float mp;
	if (fp16) mp = offset / 2000000.0f;
	else mp = offset / 4000000.0f;
	cout << "\n INFO: " << fixed << setprecision(3) << mp <<" million parameters written to `" << prefix << ".bin` !\n";
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
 
bool CNNNetwork::CheckAndPrune(const char* weights_file, float c_threshold, float w_threshold) {
	if(!weights_file) return false;
	if (!weights_pool.Load(weights_file)) {
		cerr << "Error : Load " << weights_file << " failed !\n";
		return false;
	}

	for (int i = 0; i < layers.size(); i++) {
		Layer* l = layers[i];
		for (int j = 0; j < l->modules.size(); j++) {
			InferenceModule* m = l->modules[j];
			cout << " Checking " << m->Name() << "...\n";
			if (!m->CheckRedundantChannels(c_threshold,w_threshold)) {
				cerr << " Error : CheckRedundantChannels failed at " << m->Name() << "!\n";
				return false;
			}
		}
	}
	bool pruned = false;
	for (int i = 0; i < layers.size(); i++) {
		Layer* l = layers[i];
		for (int j = 0; j < l->modules.size(); j++) {
			InferenceModule* m = l->modules[j];
			cout << " Prune " << m->Name() << "...\n";
			int saved_channels = m->GetOutputChannels();
			if (!m->Prune()) {
				cerr << " Error : Prunning failed at " << m->Name() << "!\n";
				return false;
			}
			if (saved_channels != m->GetOutputChannels()) pruned = true;
		}
	}
	if(pruned){
		string s(weights_file);		
		if (weights_pool.Save(replace_extension(s, ".prune.rweights"))) {
			cout << "Successfully save pruned model to " << s << "!\n";
		}
	}
	else {
		cout << " No prunable weights in " << weights_file << endl;
	}
	return true;
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
		cout << " Processing " << filename << " ...\n";
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

	cout << " INFO: output written to predictions/!\n";
	
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
	bool primary; // primary prediction of one gt
	//PredResultType result;
};
bool compare_confidence(PredictionInfo p1, PredictionInfo p2) {
	return p1.confidence > p2.confidence;
}
bool CNNNetwork::Eval(bool all ) {
	if (GetAppConfig().GetDatasetCount() == 0) {
		cerr << " *** No dataset definitions in the configuration! *** \r\n";
		return false;
	}
	if (input) delete[]input;
	
	for (int i = 0; i < GetAppConfig().GetDatasetCount(); i++) {
		vector<int> gts_by_cls;
		for (int c = 0; c < classes.size(); c++) {			 
			gts_by_cls.push_back(0);
		}
		
		vector<PredictionInfo> performance_data;
		const Dataset* ds = GetAppConfig().GetDataSet(i);
		cout << "  Processing dataset `" << ds->GetName() <<"` ";
		Image img;
		for (size_t j = 0; j < ds->GetSize(); j++) {
			string filename = ds->FilenameAt(j);
			cout << ".";
			//cout << "   Loading `" << filename << "`...\n";
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
			for (auto& gt : gts) {
				gts_by_cls[gt.class_id] ++;
			}
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
				auto& det = detections.at(d);				 
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
				pdi.primary = (pdi.gt_index != -1 && det.class_id == gts[pdi.gt_index].class_id);
				if (pdi.primary) {
					for (unsigned int i = prev_predicts; i < performance_data.size(); i++) {
						PredictionInfo& prev_p = performance_data[i];
						if (prev_p.gt_index == pdi.gt_index) {
							// for one gt, we treat the prediction with the best_iou as primary prediction
							if (prev_p.best_iou >= pdi.best_iou) {
								pdi.primary = false;
							}
							else {
								prev_p.primary = false;
							}
						}
					}
				}
				performance_data.push_back(pdi);
			}

			input = nullptr;
		}
		cout << "\n  " << ds->GetSize() << " files processed.\n  Configuration: Confidence Threshold: "
			<< fixed << setprecision(2) << GetAppConfig().ThreshHold() << ", NMS Threshold: " << GetAppConfig().NMSThreshold() << endl; ;
		// sort result by confidence desc
		sort(performance_data.begin(), performance_data.end(), compare_confidence);
		vector<float> metric_ious = { 0.5f, 0.75f };
		if (all) {
			metric_ious.clear();
			for (float metric_iou = 0.5f; metric_iou <= 0.95f; metric_iou += 0.05f) {
				metric_ious.push_back(metric_iou);
			}
		}
		array<float, 11> prs;
		for(float metric_iou : metric_ious){
			stringstream ss;
			cout << "\n  *** Under IoU " << fixed << setprecision(2) << metric_iou << " *** \n";
			float mAP = 0.0f; 
			float cur_p = 0.0f, cur_r = 0.0f;
			for (int c = 0; c < classes.size(); c++) {
				float acc_TP = 0.0f, acc_FP = 0.0f, avr_iou = 0.0f;
				for (auto& pr : prs) {
					pr = 0.0f;
				}
				for (auto& pdi : performance_data) {
					if (pdi.class_id != c) continue;
					if (pdi.primary && pdi.best_iou >= metric_iou) {
						acc_TP += 1.0;
						avr_iou += pdi.best_iou; 
					}
					else
						acc_FP += 1.0f;
					cur_p = acc_TP / (acc_TP + acc_FP);
					cur_r = (gts_by_cls[c] > 0) ? acc_TP / gts_by_cls[c] : 1.0f;
					int index = (int)floorf(cur_r * 10.0f);
					if (prs[index] < cur_p)
						prs[index] = cur_p;
				}
				//2021.01.25 fix
				for (int i = 0; i < 10; i++) {
					for (int j = i + 1; j < 11; j++) {
						if (prs[j] > prs[i]) prs[i] = prs[j];
					}
				}
				float ap = 0.0f; 
				 
				for (int i = 0; (i < 11) && (prs[i] != 0.0f); i++) { 
					if (i < 10 && prs[i] < prs[i + 1])
						prs[i] = prs[i + 1];
					ap += prs[i]; 
				}
				ap /= 0.11f;
				avr_iou /= (acc_TP > 0) ? acc_TP : 1.0f;
				cout << "\n  AP:" << ap << "%, R:" <<  (acc_TP * 100 / gts_by_cls[c]) << "%, P:" << (cur_p * 100.0f) << "%, IoU:" << setprecision(4) << avr_iou  ;
				string split1(90, '=');
				ss << fixed << setprecision(2) << "  " << split1 << "\n  R:"  ;
				for (float r = 0.0f; r < 1.1f; r += 0.1f) {
					ss << r << "    ";
				}
				string split2(90,'-'); 
				ss << "\n  " << setprecision(4) << split2 << "\n  P:";
				for (int i = 0; i < 11 ; i++) {
					ss << prs[i] << "  ";
				}
				ss << endl;
				mAP += ap;
			}
			if (classes.size() > 1) {
				mAP /= classes.size();
				cout << ", mAP:" << setprecision(2) << mAP << "%\n";
			}
			else {
				float F1score = (2.0f * cur_p * cur_r) / (cur_p + cur_r);
				cout << ", F1:" <<  F1score << "\n";
			}
			cout << ss.str();
		}
	} 
	return true;
}
