#include "stdafx.h"


#include "config.h"
#include "network.h"
#include "param_pool.h"
#include <direct.h>
using namespace std;

static const char* get_file_name(const char* path) {
	const char* f1 = strrchr(path, '/');
	const char* f2 = strrchr(path, '\\');
	if (f1 > f2)
		return f1 + 1;
	else if (f2 > f1)
		return f2 + 1;
	return path;
}
 
bool network_train(const char* data_definition, const char*  network_definition, 
	const char* weights_path, bool restart) {

	cout << "\n Loading application configuration `" << data_definition << "` ... ";
	
	if (!GetAppConfig().Load(data_definition)) {
		cerr << "Load configuration file `" << data_definition << "` failed!\n";
		return false;
	}
	cout << " Done!\n Loading network configuration `" << network_definition << "` ... ";
	CNNNetwork network;
	if (!network.Load(network_definition)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << network_definition << "` failed!\n";
		return false;
	}
	cout << " Done !\n Loading parameters from `" << weights_path << "`... ";
	
	if (*weights_path && !network.weights_pool.Load(weights_path)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << weights_path << "` failed!\n";
	}
	if (GetAppConfig().UpdatePolicy() == Adam) {
		string adam_weights_path(weights_path);
		replace_extension(adam_weights_path, ".adam.rweights");
		network.adam_weights_pool.Load(adam_weights_path.c_str());
	}
	cout << " Done !\n";
	if (!cuDNNInitialize()) return false;
	if (!network.Train(restart)) {
		cerr << "Train failed!\n";
		cuDNNFinalize();
		return false;
	}
	cuDNNFinalize();
	return true;

}
bool network_test(const char* data_definition, const char*  network_definition, const char* weights_path) {
	return false;
}
bool detect_image(const char* data_definition, const char*  network_definition,
	const char* weights_path, const char* image_file, const char* data_type, float threds_hold , const char* output_file) {
	cout << "\n Loading application configuration `" << data_definition << "` ... ";

	/*if (!GetAppConfig().Load(data_definition,2)) {
		cerr << "Load configuration file `" << data_definition << "` failed!\n";
		return false;
	}*/
	cout << " Done!\n Loading network configuration `" << network_definition << "` ... "; 
	cudnnDataType_t t = CUDNN_DATA_DOUBLE;
	if (data_type && *data_type) {
		string s(data_type);
		if (s == "FP32")
			t = CUDNN_DATA_FLOAT;
		else if (s == "FP16")
			t = CUDNN_DATA_HALF;
		else {
			cerr << " Warning: unrecognized data type `" << data_type << "`!\n";
		}		
	} 
	CNNNetwork network;
	if (!network.Load(network_definition,t)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << network_definition << "` failed!\n";
		return false;
	} 
	cout << " Done !\n Loading parameters from `" << weights_path << "... ";

	if (!network.weights_pool.Load(weights_path)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << weights_path << "` failed!\n";

		return false;
	}
	cout << " Done !\n";
	if (!cuDNNInitialize()) return false;
	bool ret = network.Detect(image_file, threds_hold, output_file);
	cuDNNFinalize();
	return ret;
}
bool detect_video(const char*  network_definition, const char* weights_path, const char* video_file) {
	return false;
}
bool convert_openvino(const char*  network_definition, const char* weights_path, const char* data_type, const char* ouput_dir ,const char* name ) {
 
	cudnnDataType_t t = CUDNN_DATA_DOUBLE;
	if (data_type && *data_type) {
		string s(data_type);
		if (s == "FP32")
			t = CUDNN_DATA_FLOAT;
		else if (s == "FP16")
			t = CUDNN_DATA_HALF;
		else {
			cerr << " Warning: unrecognized data type `" << data_type << "`! Use default data type defined in network configuration.\n";
		}
	}

	string dir(ouput_dir);
	if (dir.empty()) {
		dir = network_definition;
		get_dir_from_full_path(dir);
	}
	struct stat s = { 0 };
	stat(dir.c_str(), &s);
	if (0 == (s.st_mode & S_IFDIR)) {
		if (_mkdir(dir.c_str())) {
			cerr << "Error: Failed to create directory `" << dir << "` for output! \n";
			return false;
		}
	}
	CNNNetwork network;
	if (!network.Load(network_definition, t)) {
		cerr << "Error: Cannot load network definition file " << network_definition << endl;
		return false;
	}
	if (!network.weights_pool.Load(weights_path)) {
		cerr << "Error: Cannot load weights file " << weights_path << endl;
		return false;
	}
	string s_name(name);
	if (s_name.empty()) {
		s_name = get_file_name(network_definition);
		replace_extension(s_name, ".ir");
	}
	int l_index = 1;
	if (!network.OutputIRModel(dir, s_name, l_index)) {
		cerr << "Error: Create Failed!\n";
		return false;
	}
	cout << "INFO: successfully reated IR model !\n";
	return true;
}
const char data_def_message[] = "Required. dataset definition(in xml)\n";
const char network_def_message[] = "Required. network definition(in xml)\n";
const char weights_message[] = "Required in train and detect mode";
const char input_message[] = "Required. Path to input file.\n";
const char output_message[] = "Required. Path to output file.\n";
const char darknet_message[] = "Required. path to darket cfg file.\n";

struct ArgDef {
	const char* prefix;
	const char* param;
	bool exists;
	const char* hint;
};
ArgDef defs[] = {
	{ "-d", "config.xml", false, data_def_message },
	{ "-n", "", false, network_def_message },
	{ "-w", "", false, weights_message },
	{ "-i", "", false, input_message },
	{ "-o", "", false, output_message },
	{ "-c","",  false, darknet_message },
	{ "-t","", false, "data type: FP32 or FP16"},
	{ "-restart", "false",false, ""},
	{ "-threshold","0.7",false,""}
}; 

static void parse_cmd_line(int argc, char* argv[]) {
	int arg_def_cnt = sizeof(defs) / sizeof(ArgDef);
	int i = 0;
	while(i < argc ){ 
		for (int j = 0; j < arg_def_cnt; j++) {
			if (strcmp(argv[i], defs[j].prefix) == 0) {
				defs[j].exists = true;
				if (++i < argc) {
					defs[j].param = argv[i];
				}
				break;
			}
		}
		i++;
	}
}

int main(int argc, char* argv[]) { 
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif	 
	/*float tests[10];
	float pred = 0.0;
	for (int i = 0; i < 10; i++) {
		pred += 0.1f;
		tests[i] = (1.0f - pred) * focal_loss_delta(pred,0.5f, 2.0f);
		
	} */
	cudaError_t err = cudaGetDeviceCount(&gpu_device_cnt);
	cout << " " << gpu_device_cnt << " GPU detected." << endl << endl;
	if (argc < 3) {
		cerr << "Insufficient arguments! \n Usage :" << endl;
		cerr << "  \"" << argv[0] << "\"  detector train <data-file> <config-file> <weights-file>" << endl;
	}
	if (gpu_device_cnt == 0) {
		cerr << "CUDA support is required. \n To run on a computer with CUDA GPU." << endl;
		return -1;
	}
	
	DisplayGPUs(); 
	if (gpu_device_cnt > 0) {
		if (cudaSuccess != cudaSetDevice(0)) {
			cerr << "Error: Set Device failed!" << endl;
			return 1;
		}
	}
 
	const char* exe = get_file_name(argv[0]);
	if (argc < 4) {
		show_usage(exe);
		return 1;
	}

	const char* command = argv[1];
	argc--;
	argv ++;
	parse_cmd_line(argc, argv);
	const char* FLAGS_d = defs[0].param;
	const char* FLAGS_n = defs[1].param;
	const char* FLAGS_w = defs[2].param;
	const char* FLAGS_i = defs[3].param;
	const char* FLAGS_o = defs[4].param;
	const char* FLAGS_c = defs[5].param;
	const char* FLAGS_t = defs[6].param; 

	bool ret = false;
	
	
	if (strcmp(command, "train") == 0) {
		ret = network_train(FLAGS_d, FLAGS_n, FLAGS_w, defs[7].exists || defs[7].param == "true");
	}
	else if (strcmp(command, "test") == 0) {
		ret = network_test(FLAGS_d, FLAGS_n, FLAGS_w);
	}
	else if (strcmp(command, "detect") == 0) {
		float t = defs[8].exists ?  atof(defs[8].param) : 0.7;
		ret = detect_image(FLAGS_d, FLAGS_n, FLAGS_w, FLAGS_i,FLAGS_t , t , FLAGS_o);
	}
	else if (strcmp(command, "demo") == 0) {
		ret = detect_video(FLAGS_n, FLAGS_w, FLAGS_i);
	}
	else if (strcmp(command, "wconv") == 0) {
		CNNNetwork network;
		ret = network.weights_pool.TransformDarknetWeights(FLAGS_c, FLAGS_w, FLAGS_o); 
	}
	else if (strcmp(command, "openvino") == 0) {
		ret = convert_openvino(FLAGS_n, FLAGS_w, FLAGS_d, FLAGS_o, FLAGS_t);
	}
	else {
		show_usage(exe);
	}
	if(ret) return 0; 
	return 1;
}