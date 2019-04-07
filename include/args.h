#pragma once
#include <google/gflags/gflags.h>
#include <iostream>
using namespace std;

const char data_def_message[] = "Required. dataset definition(in xml)\n";
const char network_def_message[] = "Required. network definition(in xml)\n";
const char weights_message[] = "Required in train and detect mode";
const char input_message[] = "Required. Path to input file.\n";
const char output_message[] = "Required. Path to output file.\n";
const char darknet_message[] = "Required. path to darket cfg file.\n";

DEFINE_string(d, "", data_def_message);
DEFINE_string(n, "", network_def_message);
DEFINE_string(w, "", weights_message);
DEFINE_string(i, "", input_message);
DEFINE_string(o, "", output_message);
DEFINE_string(c, "", darknet_message);
/**
* \brief This function show a help message
* 
* Usage:
* 
* RQNet train -d data-def.xml -n net-def.xml [-w weights.pb]
* RQNet test -d data-def.xml -n net-def.xml -w weights.pb`
* RQNet detect -n net-def.xml -w weights.pb -i image-to-detect.jpg
* RQNet demo  -n net-def.xml -w weights.pb [-i video-to-detect.mp4]
* RQNet wconv -c darknet-net-def.cfg -i darknet-weights.weights -o darknet
*/
static void showUsage(const char* bin) {
	cout <<  "\n\nUsage : " << bin << " train|test|detect|demo|wconv [options]\n\n  Options \n\n";
	cout << "  To train a network:\n    " << bin << " train -d <path/to/data/defintions> -n <path/to/network/defintion> [-w <path/to/weights>]\n";
	cout << "\n        weights file is .pb file. If weights file is not given, then a random set of weighs are initialized.\n\n\n";
	cout << "  To test a network: \n    " << bin << " test  -d <path/to/data/defintions> -n <path/to/network/defintion> -w <path/to/weights>\n\n\n";
	cout << "  To detect objects in image:\n    " << bin << " detect -n <path/to/network/defintion> -w <path/to/weights> -i <path/to/image>\n\n\n";
	cout << "  To detect objects in video:\n    " << bin << " detect -n <path/to/network/defintion> -w <path/to/weights> [-i <path/to/vedio>]\n";
	cout << "\n       If input file is not given, then use a camera.\n\n\n";
	cout << "  To convert .weights file to .pb files:\n    " << bin << " detect -c <path/to/darknet/network/config> -i <path/to/darknet/weights> [-o <path/to/output>]\n\n\n";
	cout << " *** ATTENSION ***\n\n";
	cout << " This program is running only with CUDA support!\n\n";


}