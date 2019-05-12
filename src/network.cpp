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
 
CNNNetwork::CNNNetwork() {
	data_type = CUDNN_DATA_FLOAT; 
	data_format = CUDNN_TENSOR_NCHW;
	//workspace = nullptr;
	//workspace_size = 0;
	def_actvation = "leaky";
	truths = nullptr;
	input = nullptr;
	cur_iteration = 0;
	detection_layers = 2;
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
	int new_x = (int)(data.x * input_width);
	int new_y = (int)(data.y * input_height);

	int old_x, old_y;
	float threshold = 0.5f;// GetAppConfig().NMSThreshold();
	while (index >= 0) {
		DetectionResult &dr = detections[index];
		Box box2(dr.x, dr.y, dr.w, dr.h);
		old_x = (int)(dr.x * input_width);
		old_y = (int)(dr.y * input_height);
		if (((new_x - old_x) * (new_x - old_x) + (new_y - old_y) * (new_y - old_y)) < 144) {
			if (data.confidence < dr.confidence) {
				return;
			}
			dr = data;
			return;
		}
		float iou = BoxIoU(box, box2);
		if (iou > threshold) {
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
	const char* temp = root->Attribute("name");
	if (temp) {		
		name = temp;
	}
	else {
		name = file_part(filename);
		replace_extension(name, "");
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
				float w = anchorElement->FloatAttribute("width", 0.0f) / 416.0f;
				float h = anchorElement->FloatAttribute("height", 0.0f) / 416.0f;
				if (w > 0.0 && h > 0.0) {
					anchors.push_back(pair<float, float>(w, h));
				}
				anchorElement = anchorElement->NextSiblingElement();
			}
		}
	}
	else {	

			float s_min = 0.2, s_max = 0.9;
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
	string weights_file;
	float avg_loss = -1.0;  
	
	int input_len = mini_batch * input_channels * input_width * input_height * sizeof(float);
	if(!current_training_rotates) current_training_rotates = New RotateType[mini_batch + 1];

 
	while (!GetAppConfig().IsLastIteration(cur_iteration)) {
		loss = 0.0f;
		training_batch = 0;
		true_positives = 0;
		false_positives = 0;
		cur_iteration++;
		clock_t start_clk = clock(); 
		int total_objects = 0;
		for (int i = 0; i < GetAppConfig().GetSubdivision(); i++, total_images += mini_batch) {
			
			current_training_files.clear();
			//cout << "\nSubdivision " << i << ": loading data ... ";
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
			cout << endl;
			if (!Backward()) return false;   
		}
		
		loss /= training_batch;
		if (avg_loss < 0)
			avg_loss = loss;
		else
			avg_loss = avg_loss * 0.9 + loss * 0.1;
		int ms = (clock() - start_clk) * 1000 / CLOCKS_PER_SEC;
		int total_recalls = false_positives + true_positives;
		if (0 == total_recalls) total_recalls = 1;
		float lr = GetAppConfig().GetCurrentLearningRate(cur_iteration);
		 
		float rc = true_positives * 100.0f / total_objects;
		float pr = true_positives * 100.0f / total_recalls;
		float ap = rc * pr * 0.01f;
		char info[300];
		sprintf(info, "\n >> It %05d | Loss: %.4f, Avg-Loss: %.4f, Recall: %.2f%%, Precision: %.2f%%, ", cur_iteration,
			loss, avg_loss, rc, pr);
		cout <<info <<"Rate: " << lr << ", " << (ms * 0.001) << "s, total " << total_images << " images.\n\n";
		 
		ofstream ofs(filename,ios::app);
		if (ofs.is_open()) {
			//sprintf(info, "")
			ofs << cur_iteration << ",\t" << input_width  << ",\t" << input_height  << ",\t" <<  lr << ",\t"
				<< fixed << setprecision(2) << loss << ",\t"  << avg_loss << ",\t" << rc << ",\t" << pr << ",\t" << ap << endl;
			ofs.close();
		}
		
		for (auto l : layers) {
			if (!l->Update(lr)) return false;
		} 
		if (GetAppConfig().GetWeightsPath(cur_iteration, weights_file)) {
			if (weights_pool.Save(weights_file.c_str(), cur_iteration)) {
				cout << " INFO: Save weights to `" << weights_file << "` ! \n";
			}
			if (GetAppConfig().UpdatePolicy() == Adam) {
				replace_extension(weights_file, ".adam.rweights");
				if (!adam_weights_pool.Save(weights_file.c_str(), cur_iteration)) {
					cerr << " Warning: Save adam weights to `" << weights_file << " failed` ! \n";
				}
			}
			
		} 
		if (GetAppConfig().RadmonScale(cur_iteration, new_width, new_height) &&
			(new_width != input_width || new_height != input_width) ) {
			cout << "Input Resizing to " << new_width << "x" << new_height << " ...\n";
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

bool CNNNetwork::OutputIRModel(const string & dir, const string & name, int& l_index) const {
	string prefix = dir;
	if (prefix.find_first_of('\\') != prefix.length() - 1 &&
		prefix.find_first_of('/') != prefix.length() - 1)
		prefix += SPLIT_CHAR;
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
static uint8_t calc_color(double temp1, double temp2, double temp3) {

	double color;
	if (6.0 * temp3 < 1.0)
		color = temp1 + (temp2 - temp1) * 6.0* temp3;
	else if (2.0 * temp3 < 1.0)
		color = temp2;
	else if (3.0 * temp3 < 2.0)
		color = temp1 + (temp2 - temp1) * ((2.0 / 3.0) - temp3) * 6.0;
	else
		color = temp1;

	return (uint8_t)(color * 255);
}
void hsl2rgb(double hue, double sat, double light, uint8_t& r, uint8_t& g, uint8_t& b) {
	if (0.0 == sat) {
		r = g = b = (uint8_t)(light * 255); 
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
	r = calc_color(temp1, temp2, temp3);
	temp3 = hue; //for G, temp3=H
	g = calc_color(temp1, temp2, temp3);
	temp3 = hue - 1.0 / 3.0;//for B, temp3=H-1.0/3.0
	if (temp3 < 0.0)
		temp3 = temp3 + 1.0;
	b = calc_color(temp1, temp2, temp3);

}

bool CNNNetwork::Detect(const char* filename, float threshold, const char* output_file) {
	if (input_channels <= 0 || input_width <= 0 || input_height <= 0 || !filename)
		return false;

	GetAppConfig().ThreshHold(threshold);
	
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

	ForwardContext context = { false, false, false, nullptr};
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
 
	int l, r, t, b;
	
	uint8_t cr, cg, cb;
	for (int i = 0; i < (int)detections.size(); i++) {
		DetectionResult& dr = detections[i];

		const string& name = classes[dr.class_id];
		char conf_text[20];
		if (dr.confidence == 1.0f)
			strcpy(conf_text, ": 100%");
		else {
			sprintf(conf_text, ": %.2f%%", dr.confidence * 100.0f);
		}
		l = (int)((dr.x - dr.w * 0.5f) * mat.cols);
		r = (int)((dr.x + dr.w * 0.5f) * mat.cols);
		t = (int)((dr.y - dr.h * 0.5f) * mat.rows);
		b = (int)((dr.y + dr.h * 0.5f) * mat.rows); 
#if 0
		int cx = (int)(dr.x * mat.cols);
		int cy = (int)(dr.y * mat.rows);
		cv::Point pt(cx, cy);
		cv::circle(mat, pt, 4, color, 3);
#else
		if (classes.size() > 1) {
			hsl2rgb((double)(dr.class_id + 1) / (double)classes.size(), dr.confidence, 0.5, cr, cg, cb);
		}
		else {
			hsl2rgb( dr.confidence,1.0, 0.5, cr, cg, cb);
		}
		cv::Scalar color(cb,cg,cr);
		cv::Point ptTopLeft(l, t),  ptBottomRight(r, b);
		cv::rectangle(mat, ptTopLeft, ptBottomRight, color, 2);  
		ptTopLeft.y -= 15;
		//cv::putText(mat, name + conf_text, ptTopLeft, cv::FONT_HERSHEY_COMPLEX, 0.8, color, 2);
		//int baseline = 0;
		//cv::Size size =  cv::getTextSize(name + conf_text, cv::FONT_HERSHEY_COMPLEX, 0.8, 2, &baseline);

#endif
		std::cout << name << " at [" << l << ", " << r << ", " << t << ", " << b << " ] "  << conf_text << endl;
	}
	vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);
	params.push_back(90);
	if (!output_file || 0 == *output_file)
		output_file = "predictions.jpg";
	
	cv::imwrite(output_file, mat, params);

	std::cout << " INFO: output written to "<< output_file <<"!\n";
	
	return true;
}