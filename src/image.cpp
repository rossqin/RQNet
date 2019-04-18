#include "StdAfx.h"
#include "image.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
Image::Image() {
	height = 0;
	width = 0;
	channels = 0;
	data = NULL;
	gpu_data = NULL;
	normalized = false;
}
Image::Image(const char *filename) {
	height = 0;
	width = 0;
	channels = 0;
	data = NULL;
	gpu_data = NULL;
	normalized = false;
	Load(filename);
}

Image::Image(int w, int h, int c, float val) {
	normalized = false; 
	if (w > 0 && h > 0 && c > 0) {
		width = w;
		height = h;
		channels = c;
		int e = w * h * c;
		data = New float[e];
		if (0.0 == val)
			memset(data, 0, e * sizeof(float));
		else {
			for (int i = 0; i < e; i++)
				data[i] = val;
		}
	}
	else
		data = nullptr;
	gpu_data = nullptr;
}
Image::Image(int w, int h, int c, float* data_cpu) {
	normalized = true;
	if (w > 0 && h > 0 && c > 0) {
		width = w;
		height = h;
		channels = c;
		int e = w * h * c;
		data = New float[e];
		if (data_cpu)
			memcpy(data, data_cpu, e * sizeof(float));
		else {
			memset(data, 0, e * sizeof(float));
		}
	}
}
bool Image::PushToGPU() {
	int e = width * height * channels; 
	if (gpu_data) return true;
	if (0 == e || NULL == data) return false;
	if(cudaSuccess != cudaMalloc(&gpu_data, e * sizeof(float))) 
		return false;
	if (cudaSuccess != cudaMemcpy(gpu_data, data, e * sizeof(float), cudaMemcpyHostToDevice)) {
		cudaFree(gpu_data);
		gpu_data = NULL;
		return false;
	}
	delete[]data;
	data = NULL;
	return true;
}
bool Image::PullFromGPU() {
	if (data) return true;
	int e = width * height * channels;
	if (0 == e || NULL == gpu_data) return false;
	
	data = New float[e];
	cudaError_t err = cudaMemcpy(data, gpu_data, e * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(gpu_data);
	gpu_data = NULL;
	return (cudaSuccess == err);
}
void Image::Whiten() {
//TODO: not yet implmented.
}
Image::~Image() {
	if (data)
		delete[]data;
	if (gpu_data) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	}
}
bool Image::Crop(Image& result, int dx, int dy, int w, int h) {

	if (dx < 0 || dx > width || dy < 0 || dy > height) return false;
	if (dx + w > width) w = width - dx;
	if (dy + h > height) h = height - dy;
	if (w <= 0 || h <= 0) return false;

	
	if (result.data) {
		delete[] result.data;
		result.data = NULL;
	}
	if (result.gpu_data) {
		cudaFree(result.gpu_data);
		result.gpu_data = NULL;
	}

	if (!PushToGPU()) return false;
	
	result.channels = channels;
	result.height = h;
	result.width = w;
	result.normalized = normalized;
	cudaMalloc(&result.gpu_data, channels * w * h * sizeof(float));
	if (NULL == result.gpu_data) return false;
 
	size_t bytes = w * sizeof(float);

	int as = height * width;
	int ad = h * w;
	float *dest = result.gpu_data;
	float *src_base = gpu_data;
	for (int c = 0; c < channels; c++ , src_base += as) {
		for (int y = 0; y < h; y++ , dest += w) {
			float* src = src_base + (y + dy) * width + dx;
			if (cudaSuccess != cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice)) {
				return false;
			}
		}
	}
	return true;

}

Image::Image(const Image& img) {
	width = img.width;
	height = img.height;
	channels = img.channels;
	normalized = img.normalized;
	gpu_data = NULL;
	data = NULL;
	if (img.data) {
		int e = width * height * channels;
		data = New float[e];
		memcpy(data, img.data, e * sizeof(float)); 
	} 
	if (img.gpu_data) {
		int b = width * height * channels * sizeof(float);
		if (cudaSuccess == cudaMalloc(&gpu_data, b )) {
			cudaMemcpy(gpu_data, img.gpu_data, b, cudaMemcpyDeviceToDevice);
		} 
	}
}
 
const Image& Image::operator=(const Image& img) {
	width = img.width;
	height = img.height;
	channels = img.channels;
	normalized = img.normalized;
	if (data) {
		delete[]data;
		data = NULL;
	}
	if (img.data) {
		int e = width * height * channels;
		data = New float[e];
		memcpy(data, img.data, e * sizeof(float));
	} 
	if (gpu_data) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	}
	if (img.gpu_data) {
		int b = width * height * channels * sizeof(float);
		if (cudaSuccess == cudaMalloc(&gpu_data, b)) {
			cudaMemcpy(gpu_data, img.gpu_data, b, cudaMemcpyDeviceToDevice);
		}
	}
	return *this;
} 

bool hwc_uc_2_chw_float(float* dst, const uint8_t* src, int w, int h, int c, bool norm);
bool Image::Load(const char * filename, int c, bool norm ) {
 
	//
	uint8_t* io_buffer = NULL;
	size_t size = 0;
	unsigned int start_t = GetTickCount();
	cv::Mat mat = cv::imread(filename); 
	
	if (mat.empty()) {
		cerr << "\nError: Cannot load image `" << filename << "`\n";
		return false;
	}

	height = mat.rows;
	width = mat.cols;
	channels = mat.channels();
	//long t2 = GetTickCount();
	//cout << "****** stbi_load " << filename << " in " << (t2 - t1) << "ms.\n";
	size_t image_size = width * height * channels;

	// convert HWC to CHW
	if (data) {
		delete[]data;
		data = NULL;
	}
	if (gpu_data) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	} 
	cudaMalloc(&gpu_data, image_size * sizeof(float));
	if (NULL == gpu_data) { 
		return false;
	}
	uint8_t* temp = NULL;
	cudaMalloc(&temp, image_size);
	if (NULL == temp || (cudaSuccess != cudaMemcpy(temp, mat.data, image_size, cudaMemcpyHostToDevice))) {
		if (temp) cudaFree(temp);
		cudaFree(gpu_data);
		gpu_data = NULL; 
		return false;
	} 
	bool ret = hwc_uc_2_chw_float(gpu_data, temp, width, height, channels, norm);
	cudaFree(temp);
	
	normalized = norm;
	return ret;
}
bool Image::Gray(bool rgb /* = true */) {
	
	if (3 != channels  || 0 == height || 0 == width) {
		cerr << "Nothing to do with Image::Gray() \n";
		return false;
	}
	if (!PullFromGPU()) return false;
	float f[3];
	if (rgb) {
		f[0] = 0.299f; f[1] = 0.587f, f[2] = 0.114f;
	}
	else {
		f[0] = 0.2126f; f[1] = 0.7152f, f[2] = 0.0722f;
	}
	int new_size = width * height;
	float* new_data = New float[new_size];
	int index = 0;
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width; k++, index++) {
			new_data[index] = f[0] * data[index] + f[1] * data[index + new_size] + 
				f[2] * data[index + (new_size << 1)];
		}
	}
	delete[]data;
	channels = 1;
	data = new_data; 
	if (gpu_data) {
		cudaFree(gpu_data);
		gpu_data = NULL;
	}
	return true; 
}



bool Image::Save(const char* filename, int quality) {
	 
	if (!PullFromGPU()) return false;

	cv::Mat mat(height, width, CV_8UC3);  
	uint8_t* dat_buf = mat.data;
	int si = 0;
	for (int i = 0; i < channels; i++) {		
		int di = i;
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++, si++, di += channels) {
				if(normalized)
					dat_buf[di] = (uint8_t)(data[si] * 255);
				else
					dat_buf[di] = (uint8_t)data[si];
			}
		}
	}
 
	vector<int> params;
	string fname(filename);
	
	if (is_suffix(filename, ".jpg") || is_suffix(filename, ".JPG")) {
		params.push_back(cv::IMWRITE_JPEG_QUALITY);
		params.push_back(quality);
		//success = stbi_write_jpg(filename, width, height, channels, dat_buf, quality);
	}
	else if (is_suffix(filename, ".bmp") || is_suffix(filename, ".BMP")) {
		//success = stbi_write_bmp(filename, width, height, channels, dat_buf);
	}
	else if (is_suffix(filename, ".png") || is_suffix(filename, ".PNG")) {
		params.push_back(cv::IMWRITE_PNG_COMPRESSION);
		params.push_back(9);
		//success = stbi_write_png(filename, width, height, channels, dat_buf, width * channels);
	}
	else { 
		fname += ".jpg";
		params.push_back(cv::IMWRITE_JPEG_QUALITY);
		params.push_back(quality); 
	} 
	bool success = false;
	try {
		success = cv::imwrite(fname, mat, params);
		if (!success) {
			cerr << "Failed to write image `" << fname << "`\n";
			return false;
		}
	}
	catch (const cv::Exception& ex) {
		cerr << "Failed to write image `" << fname << "`, reasons: " << ex.what() << endl;
		return false;
	}
	
 
	return true;
}