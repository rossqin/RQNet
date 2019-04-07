#include "stdafx.h"

#include "args.h"
#include "config.h"
#include "network.h"
#include "param_pool.h"

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
bool network_train(const string& data_definition, const string&  network_definition, const string& weights_path) {
	
	cout << "\n Loading application configuration `" << data_definition << "` ... ";
	
	if (!GetAppConfig().Load(data_definition.c_str())) {
		cerr << "Load configuration file `" << data_definition << "` failed!\n";
		return false;
	}
	cout << " Done!\n Loading network configuration `" << network_definition << "` ... ";
	if (!GetNetwork().Load(network_definition.c_str())) {
		cout << " Failed! \n";
		cerr << "Load network file `" << network_definition << "` failed!\n";
		return false;
	}
	cout << " Done !\n Loading parameters from `" << weights_path << "... ";
	
	if (!GetParamPool().Load(weights_path.c_str())) {
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
bool network_test(const string& data_definition, const string&  network_definition, const string& weights_path) {
	return false;
}
bool detect_image(const string&  network_definition, const string& weights_path, const string& image_file) {
	return false;
}
bool detect_video(const string&  network_definition, const string& weights_path, const string& video_file) {
	return false;
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
		showUsage(exe);
		return 1;
	}

	

	const char* command = argv[1];
	argc--;
	argv ++;
	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
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
		ret = GetParamPool().TransformDarknetWeights(FLAGS_c, FLAGS_i,FLAGS_o); 
	}
	if(ret) return 0;
	showUsage(exe);
	return 1;
}