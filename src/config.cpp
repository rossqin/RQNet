#include "stdafx.h"
#include "tensor.h"
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
AppConfig::AppConfig() {
	dataset = NULL;
	stop_interation = 500000;
	restart_interation = false;

	save_input = false;
	input_dir = "network_input";

	save_weight_interval = 100 ;
	weight_file_prefix = "rq_weights_";
	out_dir = "backup/";
	momentum = 0.9f;
	decay = 0.005f;

	//da augmentation
	da_jitter = 0.0;
	da_saturation = 1.0;
	da_exposure = 1.0;
	da_hue = 0.0;

	//multi_scale

	ms_enable = false;
	ms_interval = 20; 
	batch = 32;
	subdivision = 8;

	//learning_rates ;
	lr_base = 0.001f;
	lr_burnin = 1000; 
	freezeConvParams = false;
	freezeBNParams = false;
	freezeActParams = false;

	fast_resize = true;
	small_object = true;
	update_strategy = "SGD";
	lr_step = 1;
	lr_scale = 1.0f;
	lr_power = 4.0f;
	lr_gamma = 1.0f;
}

AppConfig::~AppConfig()
{
}

bool AppConfig::Load(const char * filename, bool training) {


	tinyxml2::XMLDocument doc;
	if (XML_SUCCESS != doc.LoadFile(filename)) return false;

	XMLElement* root = doc.RootElement();
	if (!root) return false;
	small_object = root->BoolAttribute("small-object", true);
	const XMLElement*  ds = root->FirstChildElement("datasets");
	if (ds)
		fast_resize = ds->BoolAttribute("image-resize-fast", true);
	ds = ds->FirstChildElement("dataset"); 
	if (NULL == ds) return false;

	const char* dataset_name = NULL;
	if (training) {
		XMLElement* te = root->FirstChildElement("train-settings");
		if (NULL == te) return false;

		freezeConvParams = te->BoolAttribute("freeze-conv");
		freezeBNParams = te->BoolAttribute("freeze-activation");
		freezeActParams = te->BoolAttribute("freeze-batchnorm");

		dataset_name = te->GetText("dataset");
		te->QueryIntText("termination", stop_interation);

		te->QueryIntText("weights/save", save_weight_interval);
		te->QueryText("weights/output_dir", out_dir);
		te->QueryText("weights/prefix", weight_file_prefix);
		te->QueryFloatText("weights/momentum", momentum);
		te->QueryFloatText("weights/decay", decay);

		te->QueryFloatText("data_aug/jitter", da_jitter);
		te->QueryFloatText("data_aug/saturation", da_saturation);
		te->QueryFloatText("data_aug/exposure", da_exposure);
		te->QueryFloatText("data_aug/hue", da_hue);
		te->QueryIntText("batch", batch);
		te->QueryIntText("subdivision", subdivision);
		te->QueryFloatText("learning_rate/base", lr_base);
		te->QueryIntText("learning_rate/burn_in", lr_burnin);
		te->QueryFloatText("learning_rate/power", lr_power);
		te->QueryFloatText("learning_rate/gamma", lr_gamma);
		te->QueryBoolText("restart", restart_interation);
		te->QueryIntText("max_truths", max_truths);

		
		te->QueryText("update_strategy", update_strategy);

		te->QueryBoolText("save_input", save_input);
		te->QueryText("input_files_dir", input_dir);
		if (input_dir.find_last_of('/') != input_dir.length() &&
			input_dir.find_last_of('\\') != input_dir.length())
			input_dir += "/";
 
		struct stat s = { 0 };
		stat(input_dir.c_str(), &s);
		if (0 == (s.st_mode & S_IFDIR)) {
			int err = _mkdir(input_dir.c_str());
			if (err) {
				cerr << "Error: Try making directory `" << input_dir.c_str() << "` failed! \n";
				save_input = false;
			}
		}
		string str;
		if (XML_SUCCESS == te->QueryText("learning_rate/policy", str)) {
			if (str == "steps") {
				lr_policy = STEPS;
				const XMLElement* step = te->FirstChildElement("learning_rate/steps/step");
				while (step) {
					int it = step->IntAttribute("iteration", 0);
					float lr = step->FloatAttribute("rate",0.0);
					if (it > 0 && lr > 0.0) {
						lr_steps.push_back(pair<int, float>(it, lr));
					}
					step = step->NextSiblingElement();
				}

			}
			else if (str == "constant") {
				lr_policy = CONSTANT;
			}
		}


		const XMLElement* ms = te->FirstChildElement("multi_scale");
		if (ms) {
			ms->QueryBoolAttribute("enable", &ms_enable);
			ms->QueryIntAttribute("interval", &ms_interval);
			for (const XMLElement* c = ms->FirstChildElement(); c != NULL; c = c->NextSiblingElement()) {
				int w = c->IntAttribute("width", 0);
				int h = c->IntAttribute("height", 0);
				if (w > 0 && h > 0)  
					scales.push_back(pair<int, int>(w, h));
				 
			}
		}

		//TODO: create dir if not exists;  
	}
	if (dataset_name) {
		while (ds) {
			if (ds->Attribute("name", dataset_name)) {
				dataset = New Dataset(ds);
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
	if ((it > 0) && (save_weight_interval > 0) && (it % save_weight_interval) == 0) {
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
		
		os << out_dir << weight_file_prefix << it << ".rqweights";
		filename = os.str();
		return true;
	}
	return false;
	 
}

bool AppConfig::DataAugument(FloatTensor4D & image) const {
	return false;
}

float AppConfig::GetCurrentLearningRate(int iteration) const {
	float rate;
	if (iteration < lr_burnin)
		return  lr_base * pow((float)iteration / lr_burnin, lr_power);
	switch (lr_policy) {
	case CONSTANT:
		return lr_base;
	case STEP:
		return lr_base * pow(lr_scale, iteration / lr_step);
	case STEPS:
		rate = lr_base;
		for (int i = 0; i < (int)lr_steps.size(); ++i) {
			if (lr_steps[i].first > iteration) return rate;
			rate = lr_steps[i].second;
			//if(steps[i] > iteration - 1 && scales[i] > 1) reset_momentum(net);
		}
		return rate;
	case EXP:
		return lr_base * pow(lr_gamma, iteration);
	case POLY:
		return lr_base * pow(1 - (float)iteration / stop_interation, lr_power);
		//if (iteration < lr_burnin) return lr_base * pow((float)iteration / lr_burnin, power);
		//return lr_base * pow(1 - (float)iteration / max_batches, power);
	case RANDOM:
		return lr_base * pow(rand_uniform_strong(0.0, 1.0), lr_power);
	case SIG:
		return lr_base * (1.0f / (1.0f + exp(lr_gamma *(iteration - lr_step))));
	default:
		cerr << "Policy is weird!" << endl;
		return lr_base;
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
			str += "/";
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
