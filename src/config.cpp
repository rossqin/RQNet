#include "stdafx.h"
#include "cuda_tensor.h"
#include "config.h" 

#include <io.h>
#include <sys/stat.h>
#include <direct.h>

AppConfig theConfig;
AppConfig& GetAppConfig() {
	return theConfig;
}
void Dataset::ShuffleFiles() const {
	if (filenames.size() == 0) return;
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	shuffle(filenames.begin(), filenames.end(), default_random_engine(seed));
}
const char* AppConfig::LoadTrainingSection(XMLElement * root ) {

	XMLElement* te = root->FirstChildElement("train-settings");
	if (nullptr == te) return false;

	freeze_conv_params = te->BoolAttribute("freeze-conv"); 
	freeze_bn_params = te->BoolAttribute("freeze-batchnorm");

	const char* dataset_name = te->GetText("dataset");
	te->QueryIntText("termination", stop_interation);
	te->QueryBoolText("focal-loss", focal_loss);
	te->QueryIntText("weights/save", save_weight_interval);
	te->QueryText("weights/output-dir", out_dir);
	te->QueryText("weights/prefix", weight_file_prefix);
	te->QueryFloatText("weights/momentum", sgd_config.momentum);
	te->QueryFloatText("weights/decay", decay);

	te->QueryFloatText("data-augment/jitter", da_jitter);
	te->QueryFloatText("data-augment/saturation", da_saturation);
	te->QueryFloatText("data-augment/exposure", da_exposure);
	te->QueryFloatText("data-augment/hue", da_hue);
	te->QueryIntText("batch", batch);
	te->QueryIntText("subdivision", subdivision);

	te->QueryBoolText("train-background", train_bg);
	

	string str;
	te->QueryText("params-update-policy", str);
	upper(str); 
	if (str == "SGD") {
		update_policy = SGD;
	}
	else if (str == "ADAM") {
		update_policy = Adam;
	}
	else {
		cout << " Warning: invalid params-update-policy field, set to SGD\n";
		update_policy = SGD;
	}
	if (update_policy == SGD) {
		te->QueryFloatText("learning-rate/base", sgd_config.base_rate);
		
		te->QueryIntText("learning-rate/burn-in", sgd_config.burnin);
		te->QueryFloatText("learning-rate/power", sgd_config.power);
		te->QueryFloatText("learning-rate/scale", sgd_config.scale);
		te->QueryFloatText("learning-rate/gamma", sgd_config.gamma);
		if (XML_SUCCESS == te->QueryText("learning-rate/policy", str)) {
			if (str == "steps") {
				sgd_config.policy = SgdConfig::STEPS;
				const XMLElement* step = te->FirstChildElement("learning-rate/steps/step");
				while (step) {
					int it = step->IntAttribute("iteration", 0);
					float lr = step->FloatAttribute("rate", 0.0);
					if (it > 0 && lr > 0.0) {
						sgd_config.steps.push_back(pair<int, float>(it, lr));
					}
					step = step->NextSiblingElement();
				}

			}
			//TODO: implement more policies here
		}
	}
	else {
		te->QueryFloatText("learning-rate/base", adam_config.alpha);
		te->QueryFloatText("learning-rate/beta1", adam_config.beta1);
		te->QueryFloatText("learning-rate/beta2", adam_config.beta2);
		te->QueryFloatText("learning-rate/epsilon", adam_config.epsilon);
	}
	te->QueryBoolText("save-input", save_input);
	te->QueryText("input-files-dir", input_dir);
	if (input_dir.find_last_of('/') != input_dir.length() &&
		input_dir.find_last_of('\\') != input_dir.length())
		input_dir += SPLIT_CHAR;

	struct stat s = { 0 };
	stat(input_dir.c_str(), &s);
	if (0 == (s.st_mode & S_IFDIR)) {
		int err = _mkdir(input_dir.c_str());
		if (err) {
			cerr << "Error: Try making directory `" << input_dir.c_str() << "` failed! \n";
			save_input = false;
		}
	} 

	const XMLElement* ms = te->FirstChildElement("multi-scale");
	if (ms) {
		ms->QueryBoolAttribute("enable", &ms_enable);
		ms->QueryIntAttribute("interval", &ms_interval);
		for (const XMLElement* c = ms->FirstChildElement(); c != nullptr; c = c->NextSiblingElement()) {
			int w = c->IntAttribute("width", 0);
			int h = c->IntAttribute("height", 0);
			if (w > 0 && h > 0)
				scales.push_back(pair<int, int>(w, h));

		}
	}
	return dataset_name;
}
const char * AppConfig::LoadTestingSection(XMLElement * root) {
	return nullptr;
}
 
AppConfig::AppConfig() {
	dataset = nullptr;
	stop_interation = 500000;
	focal_loss = true;

	save_input = false;
	input_dir = "network_input";

	save_weight_interval = 100 ;
	weight_file_prefix = "rq_weights_";
	out_dir = "backup/";

	//da augmentation
	da_jitter = 0.0;
	da_saturation = 1.0;
	da_exposure = 1.0;
	da_hue = 0.0;

	//multi_scale

	ms_enable = false;
	ms_interval = 20; 
	batch = 1;
	subdivision = 1;

	//learning_rates ;
 
	freeze_conv_params = false;
	freeze_bn_params = false; 

	fast_resize = false; 
	update_policy = SGD;	
	thresh_hold = 0.5f;
	mns_thresh_hold = 0.8f;
	decay = 0.0005f;
	train_bg = true;
}

AppConfig::~AppConfig() {
	if (dataset)
		delete dataset;
}

bool AppConfig::Load(const char * filename, int mode) { 

	tinyxml2::XMLDocument doc;
	if (XML_SUCCESS != doc.LoadFile(filename)) return false;

	XMLElement* root = doc.RootElement();
	if (!root) return false; 
	root->QueryFloatText("thresh-hold", thresh_hold);
	root->QueryFloatText("nms-thresh-hold", mns_thresh_hold);
	const XMLElement* ds = nullptr;
	if (0 == mode || 1 == mode) {
		ds = root->FirstChildElement("datasets");
		if (ds)
			fast_resize = ds->BoolAttribute("image-resize-fast", true);
		ds = ds->FirstChildElement("dataset");
		if (!ds) return false;
	}

	const char* dataset_name = nullptr;
	switch(mode){ 
	case 0: // trainning
		dataset_name = LoadTrainingSection(root);
		break;
	case 1: // testing 
		dataset_name = LoadTestingSection(root);
		break;
	case 2:
		batch = 1;
		subdivision = 1; 
		break;
	default:
		return false;
	}
	
	if (dataset_name) {
		while (ds) {
			if (ds->Attribute("name", dataset_name)) {
				dataset = New Dataset(ds);
				dataset->ShuffleFiles();
				break;
			}
			ds = ds->NextSiblingElement();
		}
	} 
	return true;
}

bool AppConfig::RadmonScale(uint32_t it, int & new_width, int & new_height) const
{
	if(scales.size() == 0 || ms_enable == false) return false;
	if (it > 0 && it % ms_interval != 0) return false;
	int index = random_gen() % scales.size();
	auto s = scales[index];
	new_width = s.first;
	new_height = s.second;
	return true;
}
 
bool AppConfig::GetWeightsPath(uint32_t it, string & filename) const {
	if ((save_weight_interval > 0) && (it % save_weight_interval) == 0) {
		struct stat s = { 0 };
		ostringstream os;
		stat(out_dir.c_str(), &s);
		if (0 == (s.st_mode & S_IFDIR)) {
			int err = _mkdir(out_dir.c_str());
			if (err) {
				cerr << "Error: Try making directory `" << input_dir.c_str() << "` failed! \n";
				return false;
			}
		}
		char ch = out_dir[out_dir.length() - 1];
		if(ch == '/' || ch == '\\')
			os << out_dir << weight_file_prefix << it << ".rweights";
		else
			os << out_dir << "/" << weight_file_prefix << it << ".rweights";
		filename = os.str();
		return true;
	}
	return false;
	 
}
 

float AppConfig::GetCurrentLearningRate(int iteration) const { 
	if (update_policy == Adam)
		return adam_config.alpha;
	float rate;
	if (iteration < sgd_config.burnin) {
		float temp =  sgd_config.base_rate * pow((float)iteration / sgd_config.burnin, sgd_config.power); 
		if (temp < 1.0e-6f) temp = 1.0e-6f;
		return temp;
	}
	switch (sgd_config.policy) {
	case SgdConfig::CONSTANT:
		return sgd_config.base_rate;
	case SgdConfig::STEP:
		return sgd_config.base_rate * pow(sgd_config.scale, iteration / sgd_config.step);
	case SgdConfig::STEPS:
		rate = sgd_config.base_rate;
		for (int i = 0; i < (int)sgd_config.steps.size(); ++i) {
			if (sgd_config.steps[i].first > iteration) return rate;
			rate = sgd_config.steps[i].second;
			//if(steps[i] > iteration - 1 && scales[i] > 1) reset_momentum(net);
		}
		return rate;
	case SgdConfig::EXP:
		return sgd_config.base_rate * pow(sgd_config.gamma, iteration);
	case SgdConfig::POLY:
		return sgd_config.base_rate * pow(1 - (float)iteration / stop_interation, sgd_config.power);
		//if (iteration < sgd_config.burnin) return sgd_config.base_rate * pow((float)iteration / sgd_config.burnin, power);
		//return sgd_config.base_rate * pow(1 - (float)iteration / max_batches, power);
	case SgdConfig::RANDOM:
		return sgd_config.base_rate * pow(rand_uniform_strong(0.0, 1.0), sgd_config.power);
	case SgdConfig::SIG:
		return sgd_config.base_rate * (1.0f / (1.0f + exp(sgd_config.gamma *(iteration - sgd_config.step))));
	default:
		cerr << "Policy is weird!" << endl;
		return sgd_config.base_rate;
	}

	return 0.0f;
}

 

Dataset::Dataset(const XMLElement * element) {
	string str;
	element->QueryText("type", str);
	if (str == "folder") {
		if (XML_SUCCESS != element->QueryText("path", str))  return;
		 
		if (str.find_last_of('/') != str.length() &&
			str.find_last_of('\\') != str.length())
			str += SPLIT_CHAR;
		string  search_path = str + "*.*";
		

		_finddata_t find_data;
		intptr_t handle = _findfirst(search_path.c_str(), &find_data);
		if (handle == -1) {
			cerr << "Error: Failed to find first file under `" << str.c_str() << "`!\n";
			return ;
		} 
		bool cont = true; 
 
		while (cont) {
			if (0 == (find_data.attrib & _A_SUBDIR)) {
				if (is_suffix(find_data.name, ".jpg") ||
					is_suffix(find_data.name, ".JPG") ||
					is_suffix(find_data.name, ".png") ||
					is_suffix(find_data.name, ".PNG") ||
					is_suffix(find_data.name, ".bmp")||
					is_suffix(find_data.name, ".BMP")
					) { 
					filenames.push_back(str + find_data.name);
				}
			}
			cont = (_findnext(handle, &find_data) == 0);
		}
		_findclose(handle);
	}
	str += "classes.txt";
	ifstream clsfile(str);
	if (clsfile.is_open()) {
		while (getline(clsfile, str)) {
			classes.push_back(str);
		}
		clsfile.close();
	}

}
