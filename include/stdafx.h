#pragma once

#ifndef _STD_AFX_H_
#define _STD_AFX_H_ 
#define _CRT_RAND_S 1
 
# pragma warning (disable:4819)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <curand_kernel.h>


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

#ifdef _WIN32  
#include <Windows.h>
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
const char* get_time_str();
const char* make_path(const char* dir, const char* base, const char* ext);
float get_next_float(const char*& str);
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

#endif
