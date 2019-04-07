#include "StdAfx.h"
#include "image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 


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
			for (int i = 0; i < e; i < e)
				data[i] = val;
		}
	}
	else
		data = NULL;
	gpu_data = NULL;
}

bool Image::PushToGPU() {
	int e = width * height * channels;
	if (0 == e ||NULL == data) return false;
	 
	if(NULL == gpu_data && 
		cudaSuccess != cudaMalloc(&gpu_data, e * sizeof(float))) 
		return false;
	if (cudaSuccess != cudaMemcpy(gpu_data, data, e * sizeof(float), cudaMemcpyHostToDevice)) {
		cudaFree(gpu_data);
		gpu_data = NULL;
		return false;
	}
	return true;
}
bool Image::PullFromGPU() {
	if (NULL == data || NULL == gpu_data) return false;
	int e = width * height * channels;
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

	if(result.data)
		delete[] result.data;
	
	result.channels = channels;
	result.height = h;
	result.width = w;
	result.data = New float[channels * w * h];
	size_t bytes = w * sizeof(float);

	for (int c = 0; c < channels; c++) {
		for (int y = 0; y < h; y++) {
			float *src_base = data + c * height * width + (y + dy) * width;
			float *dest = result.data + c * w * h + y * w;
			memcpy(dest, src_base + dx, bytes); 
		}
	}
	return true;

}

Image::Image(const Image& img) {
	width = img.width;
	height = img.height;
	channels = img.channels;
	normalized = img.normalized;
	
	if (img.data) {
		int e = width * height * channels;
		data = New float[e];
		memcpy(data, img.data, e * sizeof(float));
	}
	else
		data = NULL;
	if (gpu_data) {
		cudaFree(gpu_data);
		gpu_data = NULL;
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
	return *this;
} 
constexpr float norm_factor = 1.0 / 255;
bool Image::Load(const char * filename, int c, bool norm ) {
 
	//
	stbi_uc* io_buffer = NULL;
	size_t size = 0;
	unsigned int start_t = GetTickCount();
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		file.seekg(0, ios_base::end);
		size = file.tellg();
		if (file.bad())
			cerr << "bad status in file \n";
		//	cout << "File size:" << size << " bytes.\n";
		if (size < 128) {
			cerr << "bad file size.\n";
			return false;
		}
		io_buffer = New stbi_uc[size];
		file.seekg(0, ios_base::beg);
		file.read(reinterpret_cast<char*>(io_buffer), size);
		file.close();
	}
	//stbi_load_from_memory

	unsigned char *buff = stbi_load_from_memory(io_buffer, size, &width, &height, &channels, c);
	delete[]io_buffer;
	if (!buff) {
		cerr << "Cannot load image `" << filename << "`\nSTB Reason: " << stbi_failure_reason() << endl;

		return false;
	}
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
	data = New float[image_size];
	int di = 0;	
	for (int i = 0; i < channels; i++) {
		int si = i;
		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++, di ++, si += channels) {
				if(norm)
					data[di] = (float)buff[si] * norm_factor;
				else
					data[di] = (float)buff[si];
			}
		}
	}
	delete[]buff;
	normalized = norm;
	return true;
}
bool Image::Gray(bool rgb /* = true */) {
	if (3 != channels  || 0 == height || 0 == width) {
		cerr << "Nothing to do with Image::Gray() \n";
		return false;
	}
	float f[3];
	if (rgb) {
		f[0] = 0.299; f[1] = 0.587, f[2] = 0.114;
	}
	else {
		f[0] = 0.2126; f[1] = 0.7152, f[2] = 0.0722;
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
	 
	if (NULL == data) return false;

	uint8_t *dat_buf = New uint8_t[height * width * channels];
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
	int success = 0;
	if (is_suffix(filename, ".jpg") || is_suffix(filename, ".JPG")) {
		success = stbi_write_jpg(filename, width, height, channels, dat_buf, quality);
	}
	else if (is_suffix(filename, ".bmp") || is_suffix(filename, ".BMP")) {
		success = stbi_write_bmp(filename, width, height, channels, dat_buf);
	}
	else if (is_suffix(filename, ".png") || is_suffix(filename, ".PNG")) {
		success = stbi_write_png(filename, width, height, channels, dat_buf, width * channels);
	}
	else {
		string new_filename(filename);
		new_filename += ".jpg";
		success = stbi_write_jpg(new_filename.c_str(), width, height, channels, dat_buf, quality);
	} 
	delete []dat_buf;
	if (!success) {
		cerr << "Failed to write image `" << filename << "`\n";
		return false;
	}
 
	return true;
}