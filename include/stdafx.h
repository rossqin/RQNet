#pragma once

#ifndef _STD_AFX_H_
#define _STD_AFX_H_ 
#define _CRT_RAND_S 1
 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <algorithm>
#include <sstream>

#include <chrono>
#include <random>

#include <io.h>
#include <sys/stat.h>
#include <direct.h>

#ifdef _WIN32 
#pragma warning (disable:4819 4244)
#include <Windows.h>
#define SPLIT_CHAR '\\'
#else
#define SPLIT_CHAR '/'
#endif
#ifdef _DEBUG
#define New   new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define _CRTDBG_MAP_ALLOC
#ifdef _WIN32 
#include <crtdbg.h> 
#endif
#else 
#define New new
#endif

#include <stdint.h>


const float EPSILON = 0.00001f;
using namespace std;


int GPUBlockSize(int b = 0);
int GPUGridSize(int g = 0);
void GPUGetGridBlock(int threads, int& g, int& b);
void DisplayGPUs();
string& remove_ext(string& name);
string& trim(string& s);
bool atob(const char* str);
extern int gpu_device_cnt; 

typedef const char* pconstchar;
unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min_, float max_);
const char* get_time_str(bool standard = false);
const char* make_path(const char* dir, const char* base, const char* ext);
double get_next_float(const char*& str);
int get_next_int(const char*& str);
float* make_float_vector(int n);
float rand_scale(float s);
bool cuDNNInitialize();
void cuDNNFinalize();
cudnnHandle_t GetCUDNNHandle();
float square_sum_array(float *a, int n);
float* new_gpu_array(unsigned int elements, float fill_val);
bool is_suffix(const char* filename, const char* ext);
void dump_mem(float* data, int n);
void split_string(vector<string>& result, const string& str, char ch = ',');  
const char* replace_extension(string& str, const char* new_ext);
void show_usage(const char* bin);
bool f32_to_f16(__half* dst, const float* src, size_t n);
bool f16_to_f32(float* dst, const __half* src, size_t n);
const char* get_dir_from_full_path(string& path);
void upper(string& str);
const char* file_part(const string& path);
float focal_loss_delta(float pred, float alpha = 0.5f, float gamma = 4.0f);
bool make_sure_dir_exists(const char* path);
#define FULL_DEBUGGING 1
#define DEBUGGING_DIR "\\AI\\Data\\debugging\\RQNet\\"

#define ROTATE_TYPE_COUNT 6
enum RotateType { NotRotate, ToLeft, ToRight, HorizFlip, VertiFlip, Rotate180 };
const char* rotate_to_str(RotateType rt);
#endif
