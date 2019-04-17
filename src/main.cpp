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
 
bool network_train(const char* data_definition, const char*  network_definition, const char* weights_path) {
	
	if (!cuDNNInitialize()) return false;
	cout << "\n Loading application configuration `" << data_definition << "` ... ";
	
	if (!GetAppConfig().Load(data_definition)) {
		cerr << "Load configuration file `" << data_definition << "` failed!\n";
		return false;
	}
	cout << " Done!\n Loading network configuration `" << network_definition << "` ... ";
	if (!GetNetwork().Load(network_definition)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << network_definition << "` failed!\n";
		return false;
	}
	cout << " Done !\n Loading parameters from `" << weights_path << "... ";
	
	if (!GetParamPool().Load(weights_path)) {
		cout << " Failed! \n";
		cerr << "Load network file `" << weights_path << "` failed!\n";
		
		return false;
	}
	cout << " Done !\n";
	
	if (!GetNetwork().Train()) {
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
bool detect_image(const char*  network_definition, const char* weights_path, const char* image_file) {
	return false;
}
bool detect_video(const char*  network_definition, const char* weights_path, const char* video_file) {
	return false;
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
	const char* hint;
};
ArgDef defs[] = {
	{ "-d", "", data_def_message },
	{ "-n", "", network_def_message },
	{ "-w", "", weights_message },
	{ "-i", "", input_message },
	{ "-o", "", output_message },
	{ "-c","", darknet_message },
	{ "-t","", ""}
}; 

static void parse_cmd_line(int argc, char* argv[]) {
	int arg_def_cnt = sizeof(defs) / sizeof(ArgDef);
	int i = 0;
	while(i < argc ){ 
		for (int j = 0; j < arg_def_cnt; j++) {
			if (strcmp(argv[i], defs[j].prefix) == 0) {
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
		ret = network_train(FLAGS_d, FLAGS_n, FLAGS_w);
	}
	else if (strcmp(command, "test") == 0) {
		ret = network_test(FLAGS_d, FLAGS_n, FLAGS_w);
	}
	else if (strcmp(command, "detect") == 0) {
		ret = detect_image(FLAGS_n, FLAGS_w, FLAGS_i);
	}
	else if (strcmp(command, "demo") == 0) {
		ret = detect_video(FLAGS_n, FLAGS_w, FLAGS_i);
	}
	else if (strcmp(command, "wconv") == 0) {
		ret = GetParamPool().TransformDarknetWeights(FLAGS_c, FLAGS_i, FLAGS_o); 
	}
	else if (strcmp(command, "openvino") == 0) {
		string name(FLAGS_t);

		string dir(FLAGS_o);
		if (dir.empty()) {
			dir = FLAGS_n;
			get_dir_from_full_path(dir);
		}
		struct stat s = { 0 };
		stat(dir.c_str(), &s);
		if (0 == (s.st_mode & S_IFDIR)) {
			if (_mkdir(dir.c_str())) {
				cerr << "Error: Failed to create directory `" << dir << "` for output! \n";
				return 1;
			}
		}
		if (!GetNetwork().Load(FLAGS_n)) {
			cerr << "Error: Cannot load network definition file " << FLAGS_n << endl;
			return 1;
		}
		if (!GetParamPool().Load(FLAGS_w)) {
			cerr << "Error: Cannot load weights file " << FLAGS_w << endl;
			return 1;
		}
		
		if (name.empty()) {
			name = get_file_name(FLAGS_n);
			replace_extension(name, ".ir");
		}
		if (!GetNetwork().OutputIRModel(dir, name, strcmp(FLAGS_d, "FP16") == 0)) {
			cerr << "Error: Create Failed!\n"  ;
			return 1;
		}
		cout << "INFO: successfully reated IR model !\n";
	}
	if(ret) return 0; 
	return 1;
}