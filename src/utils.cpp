#include "stdafx.h"
 
 
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
	if (NULL == str) return false;
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
	unsigned int rnd = 0;
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
static char time_str_buf[32];
const char* get_time_str() {
	time_t n = time(NULL);
	tm ti;
	localtime_s(&ti, &n);
	sprintf_s(time_str_buf, 32, "%04d%02d%02d%02d%02d%02d",
		ti.tm_year + 1900, ti.tm_mon + 1, ti.tm_mday,
		ti.tm_hour, ti.tm_min, ti.tm_sec);
	return time_str_buf;
}
static char path[300];
const char* make_path(const char* dir, const char* base, const char* ext) {
	sprintf_s(path, 300, "%s\\%s%s", dir, base, ext);
	return path;
}
float get_next_float(const char*& str) {
	const char* p = str;
	while (*p != ',' && *p != ' '&& *p != '\t' && *p != 0)
		p++;
	float r = (float)atof(str);
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
	return 1.0 / scale;
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
	float* ret = NULL;
	size_t bytes = elements * sizeof(float);
	cudaError_t e = cudaMallocManaged(&ret, bytes);
	// Notice : param no.2 of the cudaMalloc function always indicates "bytes" not "elements
	if (e != cudaSuccess) {
		cerr << "Error: cudaMalloc ret " << e << "in new_gpu_array!\n";
		return NULL;
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