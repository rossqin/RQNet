#include "stdafx.h"
#include "tensor.h"
 


FloatTensor4D::FloatTensor4D(const FloatTensor4D & right) {

	gpu_data = NULL;
	cpu_data = NULL;
	batch = right.batch;
	channels = right.channels;
	height = right.height;
	width = right.width;
	elements_2d = right.elements_2d;
	elements_3d = right.elements_3d;
	elements = right.elements;
	bytes = right.bytes;
	cudaError_t err = cudaMalloc(&gpu_data, bytes);
	if (err != cudaSuccess) {
		cout << "Error: Try to allocate " << bytes << " bytes of GPU memory failed in FloatTensor4D.copy_constructor ! Error code : " << err << endl;
	}
	else {
		err = cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			cout << "Error: Try to copy " << bytes << " bytes of GPU memory failed in FloatTensor4D.copy_constructor! Error code : " << err << endl;
		}
	}
}

FloatTensor4D::~FloatTensor4D() {
	if (gpu_data) cudaFree(gpu_data);
	if (cpu_data) delete[] cpu_data;
}
FloatTensor4D::FloatTensor4D(bool in_gpu ) {
	bytes = 0;
	elements = 0;
	gpu_data = NULL;
	cpu_data = NULL;
	elements_2d = 0;
	elements_3d = 0;
	batch = 0;
	channels = 0;
	height = 0;
	width = 0;
	fine = true;
	order = TO_NCHW;
	data_in_gpu = in_gpu;
}
bool FloatTensor4D::Init(int b, int c, int w, int h ,TensorOrder o) {
	if (batch == b && channels == c && height == h && width == w)
		return true;
	order = o;
	if (gpu_data) {
		cudaError_t e1 = cudaPeekAtLastError();
		cudaError_t err = cudaFree(gpu_data);
		gpu_data = NULL;
		if (err != cudaSuccess) {
			cerr << "cudaFree at 0x" << hex << setw(10) << setfill('0') << (size_t)gpu_data << " return " << (int)err << endl;
		}
	}
	if (cpu_data) {
		delete []cpu_data;
		cpu_data = NULL;

	}
	batch = b;
	channels = c;
	height = h;
	width = w;
	if (order == TO_NCHW) {
		elements_2d = height * width;
		elements_3d = channels * elements_2d;
	}
	else {
		elements_2d = width * channels;
		elements_3d = height * elements_2d;
	}
	elements = batch * elements_3d;
	bytes = sizeof(float) * elements;
	if (data_in_gpu) {
		
		cudaError_t err = cudaMalloc(&gpu_data, bytes);
		if (err != cudaSuccess) {
			cout << "Error: Try to allocate " << bytes << " bytes of GPU memory failed in FloatTensor4D.constructor! Error code : " << err << endl;
			return false;
		}
		// maybe not neccesary
		cudaMemset(gpu_data, 0, bytes);
	}
	else {
		cpu_data = New float[elements];
		memset(cpu_data, 0, bytes);
	}
 
	return true;
}

bool FloatTensor4D::Release() { 
	cudaError_t err = cudaSuccess;
	if (gpu_data != NULL) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	}
	if (cpu_data != NULL) {
		delete[]cpu_data;
		cpu_data = NULL;
	}
	data_in_gpu = true;
	bytes = 0;
	elements = 0;
	
	elements_2d = 0;
	elements_3d = 0;
	batch = 0;
	channels = 0;
	height = 0;
	width = 0;
	return (err == cudaSuccess);
}
//TODO: check cpu_data
const FloatTensor4D & FloatTensor4D::operator=(const FloatTensor4D & right) { 
	if (right.data_in_gpu != data_in_gpu) {
		if (data_in_gpu)
			MoveDataToCPU();
		else
			RestoreDataFromCPU();
	}
	if (SameDememsion(right)) {
		if (data_in_gpu) {
			if (gpu_data != NULL) 
				 fine = (cudaSuccess == cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice)); 
			return *this;
		}
		else {
			memcpy(cpu_data, right.cpu_data, bytes);
			return  *this;
		}
	}
	else {
		if (gpu_data) cudaFree(gpu_data);
		gpu_data = NULL;
		batch = right.batch;
		channels = right.channels;
		height = right.height;
		width = right.width;
		order = right.order;
		elements_2d = right.elements_2d;
		elements_3d = right.elements_3d;
		elements = right.elements;
		bytes = right.bytes;
		if (right.data_in_gpu) {
			cudaMalloc(&gpu_data, bytes); 
			cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice); 
		}
		else {
			cpu_data = New float[elements];
			memcpy(cpu_data, right.cpu_data, bytes);
		}
	}
	
	return *this;
}
bool FloatTensor4D::SameDememsion(const FloatTensor4D& right) const {
	return batch == right.batch && channels == right.channels && height == right.height && width == right.width;
}
bool FloatTensor4D::Set3DData(int index, const float* src, bool src_from_cpu) {
	if (NULL == gpu_data || index < 0 || index >= batch )
		return false;
	void * dest = gpu_data + index * elements_3d;
	cudaError_t err = cudaMemcpy(dest, src, elements_3d * sizeof(float),
		src_from_cpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);

	return  err == cudaSuccess;
}
void FloatTensor4D::DumpToFile(const string& filename, int b, int c) const { 
	if (filename.length() == 0 || 0 == elements || b < 0 || c < 0 || b >= batch || c >= channels) return;

	int s = width * height;
	float* buffer = new float[s];
	memset(buffer, 0, sizeof(float) * s);
	if (order == TO_NCHW) {
		cudaMemcpy(buffer, gpu_data + (b * channels + c) * s, s * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else {

	}
	ofstream f(filename, ios::trunc);
	if (f.is_open()) {
		char temp[20];
		f << "batch " << b << ", channel " << c << endl;
		int i = 0;
		for (int y = 0; y < height; y++) {
			f << endl;
			for (int x = 0; x < width; x++,i++) {
				sprintf(temp, "%4.4f ", buffer[i]);
				f << setw(8) << temp;
			}
		}

		f.close();
	}
	delete[]buffer;
}

float* FloatTensor4D::MoveDataToCPU() {
	if (false == data_in_gpu) return cpu_data;
	if (gpu_data == NULL)  return NULL;
	
	cpu_data = New float[elements];
	cudaError_t e = cudaMemcpy(cpu_data,gpu_data, bytes, cudaMemcpyDeviceToHost);
	cudaFree(gpu_data);
	gpu_data = NULL;
	data_in_gpu = false;
	return cpu_data;
}
float* FloatTensor4D::RestoreDataFromCPU() {
	if (data_in_gpu)  return gpu_data;
	if (cpu_data == NULL) return NULL;	
	cudaMalloc(&gpu_data, bytes);
	cudaError_t e = cudaMemcpy(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice);
	delete[]cpu_data;
	cpu_data = NULL;
	data_in_gpu = true;
	return gpu_data;
}
bool FloatTensor4D::CopyDataFromCPU(void * data, size_t data_bytes, DataType data_type, uint16_t dims[4]) {
	if (NULL == MoveDataToCPU()) return false;
	if (0 == bytes) return false;
	if (data_bytes >= bytes) {
		memcpy(cpu_data, data, bytes); 
	}
	else {
		memcpy(cpu_data, data, data_bytes);	 
		for (size_t t = data_bytes / sizeof(float); t < elements; t++) {
			cpu_data[t] = rand_uniform_strong(-1.0, 1.0);
		}
	} 
	return true;
}