#include "stdafx.h"
#include <direct.h>
static int gpu_processors = 10;
static int gpu_kernels_in_mp = 128;
int gpu_device_cnt = 0;

static cudnnHandle_t cudnnHandle;

int GPUBlockSize(int b) {
	return (b > 0 && b < gpu_kernels_in_mp) ? b : gpu_kernels_in_mp;
}

int GPUGridSize(int g) {
	return (g > 0 && g < gpu_processors) ? g : gpu_processors;
}
void GPUGetGridBlock(int threads, int& g, int& b) {
	if (threads <= gpu_kernels_in_mp) {
		g = 1;
		b = threads;
	}
	else {
		g = (threads + gpu_kernels_in_mp - 1) / gpu_kernels_in_mp;
		if (g > gpu_processors) g = gpu_processors;
		b = gpu_kernels_in_mp;
	}

}
string& remove_ext(string& name) {
	size_t pos = name.find_last_of('.');
	if (pos != string::npos)
		name.erase(pos, name.length() - pos);
	return name;
}
string& trim(string& s) {
	if (s.empty())
		return s;
	s.erase(0, s.find_first_not_of(' '));
	s.erase(s.find_last_not_of(' ') + 1);
	return s;
}
bool atob(const char* str) {
	if (nullptr == str) return false;
	return (0 != strcmp(str, "0") && 0 != _strcmpi(str, "false"));
}

void DisplayGPUs() {
	cudaDeviceProp devProp;
	for (int i = 0; i < gpu_device_cnt; i++) {
		cudaError_t err = cudaGetDeviceProperties(&devProp, i);
		if (cudaSuccess == err) {
			if (0 == i) gpu_processors = devProp.multiProcessorCount;
			cout << "** GPU NO. " << i << ": " << devProp.name << endl;
			cout << "   Memory Size: " << (devProp.totalGlobalMem >> 20) << " MB." << endl;
			cout << "   Processors Count: " << devProp.multiProcessorCount << endl;
			cout << "   Shared Memory Per Block: " << (devProp.sharedMemPerBlock >> 10) << " KB" << endl;
			cout << "   Max Threads Per Block: " << devProp.maxThreadsPerBlock << endl;
			cout << "   Max Threads Per MultiProcessor: " << devProp.maxThreadsPerMultiProcessor << endl;
			cout << "   Max Warps Per MultiProcessor: " << (devProp.maxThreadsPerMultiProcessor >> 5) << endl;

		}

	}
	cout << endl << endl;
}

 

unsigned int random_gen() {
	unsigned int rnd = clock();
#ifdef _WIN32
	rand_s(&rnd);
#else
	rnd = rand();
#endif
	return rnd;
}

float random_float() {
#ifdef _WIN32
	return ((float)random_gen() / (float)UINT_MAX);
#else
	return ((float)random_gen() / (float)RAND_MAX);
#endif
}

float rand_uniform_strong(float min_, float max_) {
	if (max_ < min_) {
		float swap = min_;
		min_ = max_;
		max_ = swap;
	}
	return (random_float() * (max_ - min_)) + min_;
}
static char time_str_buf[64];
const char* get_time_str(bool standard) {
	time_t n = time(nullptr);
	tm ti;
	localtime_s(&ti, &n);
	const char* fmt = standard ? "%04d-%02d-%02d %02d:%02d:%02d" : "%04d%02d%02d%02d%02d%02d";
	sprintf(time_str_buf, fmt, ti.tm_year + 1900, ti.tm_mon + 1, ti.tm_mday, ti.tm_hour, ti.tm_min, ti.tm_sec);
	return time_str_buf;
}

static char path[300];
const char* make_path(const char* dir, const char* base, const char* ext) {
	sprintf_s(path, 300, "%s\\%s%s", dir, base, ext);
	return path;
}
double get_next_float(const char*& str) {
	const char* p = str;
	while (*p != ',' && *p != ' '&& *p != '\t' && *p != 0)
		p++;
	double r = atof(str);
	if (0 == *p)
		str = p;
	else {
		str = p + 1;
		while (*str == ',' || *str == ' ' || *str == '\t')
			str++;
	}
	return r;
}
int get_next_int(const char*& str) {
	const char* p = str;
	while (*p != ',' && *p != ' '&& *p != '\t' && *p != 0)
		p++;
	int r = atoi(str);
	if (0 == *p)
		str = p;
	else {
		str = p + 1;
		while (*str == ',' || *str == ' ' || *str == '\t')
			str++;
	}
	return r;
}
float* make_float_vector(int n) {
	float* ret = New float[n];
	memset(ret, 0, sizeof(float) * n);
	return ret;
}
float rand_scale(float s) {
	float scale = rand_uniform_strong(1, s);
	if (random_gen() % 2) return scale;
	return 1.0f / scale;
}

bool cuDNNInitialize() {
	cudnnStatus_t status = cudnnCreate(&cudnnHandle);

	if (CUDNN_STATUS_SUCCESS != status) {
		cerr << " cuDNN Initialization failed :" << status << endl;
		return false;
	}
	return true;
}

void cuDNNFinalize() {
	cudnnDestroy(cudnnHandle);
}
cudnnHandle_t GetCUDNNHandle() {
	return cudnnHandle;
}

float square_sum_array(float *a, int n) {

	float sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += a[i] * a[i];
	}
	return sum;
}
float* new_gpu_array(unsigned int elements, float fill_val) {
	float* ret = nullptr;
	size_t bytes = elements * sizeof(float);
	cudaError_t e = cudaMallocManaged(&ret, bytes);
	// Notice : param no.2 of the cudaMalloc function always indicates "bytes" not "elements
	if (e != cudaSuccess) {
		cerr << " *** Error: cudaMalloc ret " << e << " in new_gpu_array!\n";
		return nullptr;
	}
	for (unsigned int n = 0; n < elements; n++) {
		ret[n] = fill_val;
	}
	return ret;

}
bool is_suffix(const char* filename, const char* ext) {
	size_t l1 = strlen(filename);
	size_t l2 = strlen(ext);
	if (l1 < l2) return false;
	const char* s = filename + (l1 - l2);
	return 0 == strcmp(s, ext);
}
void dump_mem(float* data, int n) {
	for (int i = 0; i < n; i++) {
		if (0 == (i % 16)) cout << endl;
		cout << " " << data[i];

	}
	cout << endl;
}
void split_string(vector<string>& result, const string& str, char ch) {
	size_t off = 0;
	while (off < str.length()) {
		size_t pos = str.find(ch, off);
		if (pos == string::npos) pos = str.length();
		string t = str.substr(off, pos - off);
		trim(t);
		result.push_back(t);
		off = pos + 1;
	}
}
void parse_integers(vector<int>& result, const string& str) {
	char buffer[32];
	int pos = 0;
	const char* s = str.c_str();
	const char* end = s + str.length();
	while (s < end) {
		if (*s >= '0' && *s <= '9') {
			buffer[pos++] = *s;
		}
		else {
			buffer[pos] = 0;
			if (pos > 0) {
				result.push_back(atoi(buffer));
				pos = 0;
			}
		}
		s++;
	}
	
	if (pos > 0) {
		buffer[pos] = 0;
		result.push_back(atoi(buffer));
	}
}
const char* replace_extension(string& str, const char* new_ext) {
	size_t pos = str.find_last_of('.');
	if (string::npos == pos) {
		str += new_ext;
		return str.c_str();
	}
	str.erase(pos);
	str += new_ext;
	return str.c_str();
}
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
void show_usage(const char* bin) {
	cout << "\n\nUsage : " << bin << " train|test|detect|demo|wconv|openvino [options]\n\n  Options \n\n";
	cout << "  To train a network:\n    " << bin << " train -d <path/to/data/defintions> -n <path/to/network/defintion> [-w <path/to/weights>]\n";
	cout << "\n        weights file is .pb file. If weights file is not given, then a random set of weighs are initialized.\n\n\n";
	cout << "  To test a network: \n    " << bin << " test  -d <path/to/data/defintions> -n <path/to/network/defintion> -w <path/to/weights>\n\n\n";
	cout << "  To detect objects in image:\n    " << bin << " detect -n <path/to/network/defintion> -w <path/to/weights> -i <path/to/image>\n\n\n";
	cout << "  To detect objects in video:\n    " << bin << " demo -n <path/to/network/defintion> -w <path/to/weights> [-i <path/to/vedio>]\n";
	cout << "\n       If input file is not given, then use a camera.\n\n\n";
	cout << "  To convert .weights file to .rweights files:\n    " << bin << " wconv -c <path/to/darknet/network/config> -w <path/to/darknet/weights> [-o <dir/of/output>]\n\n\n";
	cout << " To create openvino IR models\n    " << bin << " openvino -n <path/to/RQNet/network/definition> -w <path/to/rweights> -d <FP32|FP16> -o <dir/of/output> -name <modeltitle>\n"; 
	cout << " *** ATTENSION ***\n\n";
	cout << " This program is running only with CUDA support!\n\n";


}
const char* get_dir_from_full_path(string& path) {
	size_t pos = path.find_last_of('/');
	if (pos == string::npos) pos = 0;
	size_t pos1 = path.find_last_of('\\');
	if (pos1 == string::npos) pos1 = 0;
	if (pos < pos1) pos = pos1;
	if (pos > 0) {
		path = path.substr(0, pos + 1);
		return path.c_str();
	}
	struct stat s = { 0 };
	stat(path.c_str(), &s);
	if (0 != (s.st_mode & S_IFDIR)) {
		return path.c_str();
	}
	return "";
}
void upper(string& str) {
	for (size_t i = 0; i < str.length(); i++) {
		if (str[i] >= 'a' && str[i] < 'z')
			str[i] += ('A' - 'a');
	}
}
const char* file_part(const string& path) {	 
	size_t last_pos = 0;
	for (int i = (int)path.length() - 1; i >= 0; i--) {
		if (path[i] == '/' || path[i] == '\\') {
			return path.c_str() + i + 1;
		}
	}
	return path.c_str();
}


const char* rotate_to_str(RotateType rt) {
	switch (rt)
	{
	case ToLeft:
		return "rleft";
	case ToRight:
		return "rright";
	case HorizFlip:
		return "hflip";
	case VertiFlip:
		return "vflip";
	case Rotate180:
		return "rot180";
	default:
		return "original";
	}
}

float focal_loss_delta(float pred, float alpha, float gamma) {
 
	float pt = fmaxf(pred, 0.0000001f);
	//float grad = alpha * (gamma * (pred - 1.0f) * logf(pt)* pred + (1.0f - pred)*(1.0f - pred));
	float grad = alpha * (pt - 1.0f) *(gamma * pt * logf(pt) + pt - 1.0f); //
	//float focal_grad = alpha * (pred - 1.0f) * (gamma * pred * logf(pred) + pred - 1.0f);
	return grad;
}
bool make_sure_dir_exists(const char* path) {
	struct stat s = { 0 };
	stat(path, &s);
	if (0 != (s.st_mode & S_IFDIR)) {
		return true;
	}
	return 0 == mkdir(path);
}