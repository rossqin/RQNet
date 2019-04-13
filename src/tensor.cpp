#include "stdafx.h"
#include "tensor.h"
 


FloatTensor4D::FloatTensor4D(const FloatTensor4D & right) {

	gpu_data = NULL; 
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
}
FloatTensor4D::FloatTensor4D() {
	bytes = 0;
	elements = 0;
	gpu_data = NULL; 
	elements_2d = 0;
	elements_3d = 0;
	batch = 0;
	channels = 0;
	height = 0;
	width = 0;
	fine = true;
	order = TO_NCHW; 
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
		
	cudaError_t err = cudaMalloc(&gpu_data, bytes);
	if (err != cudaSuccess) {
		cout << "Error: Try to allocate " << bytes << " bytes of GPU memory failed in FloatTensor4D.constructor! Error code : " << err << endl;
		return false;
	}
	// maybe not neccesary
	cudaMemset(gpu_data, 0, bytes);
 
	return true;
}

 

bool FloatTensor4D::Release() { 
	cudaError_t err = cudaSuccess;
	if (gpu_data != NULL) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	} 
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
	 
	if (SameDememsion(right)) {
	 
		if (gpu_data != NULL) 
				fine = (cudaSuccess == cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice)); 
		return *this;
		 
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
		cudaMalloc(&gpu_data, bytes);
		cudaMemcpy(gpu_data, right.gpu_data, bytes, cudaMemcpyDeviceToDevice);
	 
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
bool FloatTensor4D::Get3DData(int index, float * out, bool to_cpu) const {
	if (NULL == gpu_data || index < 0 || index >= batch || NULL == out)
		return false;
	void * src = gpu_data + index * elements_3d;
	cudaError_t err = cudaMemcpy(out, src, elements_3d * sizeof(float),
		to_cpu ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
	return  err == cudaSuccess;
}
bool FloatTensor4D::Concat(int b, int c, const float * src, int src_c, int src_w, int src_h) {
	if (b < 0 || b >= batch ||  c < 0 || NULL == src || NULL == gpu_data) return false;
	float* dest = gpu_data+ b * elements_3d + c * elements_2d; 
	if (src_w == width && src_h == height) {
		return add_in_gpu(dest, src, src_c * elements_2d);
	}
	else { 
		
		int h = min(src_h, height);
		int w = min(src_w, width);
		if(TO_NCHW == order){			
			for (int i = 0; i < src_c; i++) {
				for (int j = 0; j < h; j++) {
					if(!add_in_gpu(dest, src, w)) return false; 
					src += src_w;
					dest += width;
				}
			} 
		}
		else {			
			for (int i = 0; i < h; i++) { 
				if (!add_in_gpu(dest, src, src_c * w)) return false; 
				src += src_w * src_c;
				dest += width * src_c;
			}
		}
	}
	return true;
}
void FloatTensor4D::DumpToFile(const string& filename, int b, int c) const { 
	if (filename.length() == 0 || 0 == elements ) return;
	if (b < 0 || c < 0) {
		char* temp = CopyToCPU();
		char line[100];
		ofstream f(filename, ios::trunc);
		int index = 0;
		for (b = 0 ; b < batch; b++) {
			int n = sprintf(line, "b: %d\n", b);
			f.write(line, n); 
			for (int c = 0; c < channels; c++) {
				for (int i = 0; i < width * height ; i++) {
					n = sprintf(line, "%e ", temp[index]++);
					f.write(line, n);
				}
				//f.write("\n", 1);
			}
			f.write("\n", 1);
		}
		f.close();
		delete[]temp;
		return;
	}
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
				sprintf(temp, "%.6f ", buffer[i]);
				f << setw(10) << temp;
			}
		}

		f.close();
	}
	delete[]buffer;
}
#if 0
float* FloatTensor4D::MoveDataToCPU() {
	if (false == data_in_gpu) return cpu_data;
	if (gpu_data == NULL)  return NULL;
	
	cpu_data = New float[elements];
	cudaError_t e = cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
	e = cudaFree(gpu_data);
	gpu_data = NULL;
	data_in_gpu = false;
	return (e == cudaSuccess) ? cpu_data : NULL;
}
float* FloatTensor4D::RestoreDataFromCPU() {
	if (data_in_gpu)  return gpu_data;
	if (cpu_data == NULL) return NULL;	
	gpu_data = NULL;
	cudaError_t e = cudaMalloc(&gpu_data, bytes);
	cout << ((unsigned char*)cpu_data)[bytes - 1] << endl;
	e = cudaMemcpy(gpu_data, cpu_data, bytes, cudaMemcpyHostToDevice);
	delete[]cpu_data;
	cpu_data = NULL;
	data_in_gpu = true;
	return (e == cudaSuccess ) ? gpu_data : NULL ;
}
#endif
bool FloatTensor4D::CopyDataFromCPU(void * data, size_t data_bytes, DataType data_type, uint16_t dims[4]) {
	if (0 == bytes) return false;
	if (data_bytes >= bytes) {
		return cudaSuccess ==  cudaMemcpy(gpu_data, data, bytes,cudaMemcpyHostToDevice); 
	}
	else {
		if (cudaSuccess != cudaMemcpy(gpu_data, data, data_bytes, cudaMemcpyHostToDevice))
			return false;
		int to_go = (bytes - data_bytes) / sizeof(float);
		float* cpu_data = New float[to_go];
		for (int t = 0; t < to_go; t++) {
			cpu_data[t] = rand_uniform_strong(-1.0, 1.0);
		}
		cudaError_t e = cudaMemcpy(gpu_data, cpu_data, to_go * sizeof(float), cudaMemcpyHostToDevice);
		delete[]cpu_data;
		return e == cudaSuccess;
	} 
	return true;
}
char* FloatTensor4D::CopyToCPU() const {
	if (0 == bytes) return nullptr;
	char* buffer = New char[bytes];
	if (cudaSuccess != cudaMemcpy(buffer, gpu_data, bytes, cudaMemcpyDeviceToHost)) {
		delete[] buffer;
		return nullptr;
	}
	return buffer;
}
bool FloatTensor4D::SaveBatchData(const string & filename, int b) { 
	ofstream of(filename.c_str(), ios::binary | ios::trunc);
	int writing_bytes;
	float* buffer;
	if (b < 0 || b >= batch) {
		writing_bytes = bytes;
		buffer = gpu_data;
	}
	else {
		writing_bytes = elements_3d * sizeof(float);
		buffer = gpu_data + elements_3d * b;
	}

	char* data = new char[writing_bytes];
	
	if (cudaSuccess != cudaMemcpy(data, buffer, writing_bytes, cudaMemcpyDeviceToHost)) {
		delete[]data;
		return false;
	}
	if (of.is_open()) {
		of.write(data, writing_bytes);
		of.close();
	}
	delete[]data;
	return true;
}