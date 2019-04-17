#pragma once
#include "image.h"
struct ImageLabelParseInfo {
	const char* file_name;
	int max_boxes;
	RotateType rotate;
	bool small_object;
	int crop_l;
	int crop_r;
	int crop_t;
	int crop_b;
	int orig_w;
	int orig_h;
	int actual_w;
	int actual_h;
	float hue;
	float sat;
	float expo;
	int classes;
	float* image_data;
	ObjectInfo* truth_data;
	int channels;
	int image_size;
	bool failed;
	int thread_index;
};
typedef char* pstr_t;

class DataLoader {
protected: 
	int start_index;
	//bool ImagePostLoad(Image& image, ImageLabelParseInfo& info);
	//bool LoadImageLabel(const char* filename, ObjectInfo* labels, const ImageLabelParseInfo& info);
public:  
	DataLoader() { start_index = 0; }
	~DataLoader() {}

	bool MiniBatchLoad(float* input, ObjectInfo* truth_data, int channels, int mini_batch, int width, int height);


};

