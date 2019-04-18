#pragma once
 
#define ROTATE_TYPE_COUNT 6
enum RotateType { NotRotate, ToLeft, ToRight, HorizFlip, VertiFlip, Rotate180 };
class FloatTensor4D;
class Image {
protected:
	bool normalized;
	int channels;
	int width;
	int height;
	float* data;
	float* gpu_data;
	
public: 
	Image();
	Image(const char* filename);
	Image(int w, int h, int c, float val= 0.0);
	Image(int w, int h, int c, float* data_cpu);
	Image(const Image& img);

	const Image& operator=(const Image& img);
	bool Load(const char* filename, int c = 3, bool norm = true);
	bool Save(const char* filename, int quality = 100);
	bool ResizeTo(int w, int h, bool fast = false, float center_ratio = 0.4);
	//float* LoadLabels(const char* filename, int& boxes);
	bool Distort(float hue, float sat, float val);
	bool RGB2HSV(float hue = 0.0f, float sat = 1.0f, float val = 1.0f);
	bool HSV2RGB();
	bool Crop(Image& result, int dx, int dy, int w, int h);
	bool Rotate(RotateType t); 

	void Whiten();
	virtual ~Image()  ;
	inline int GetHeight() const { return height; }
	inline int GetWidth() const { return width; }
	inline int GetChannels() const { return channels; }
	inline const float* GetData() const { return data; }
	inline const float* GetGPUData() const { return gpu_data; }
	bool Gray(bool rgb = true); 
	bool PushToGPU();
	bool PullFromGPU();
};


