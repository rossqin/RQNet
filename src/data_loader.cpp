#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"
#include <thread>
#include <memory>
#include <array>
//#define _USE_MULTI_THREAD_ 1
int post_load_rot[16] = { 3,4,1,5,0,1,0,1,3,1,4,0,3,4,4,1 };
int rot_index = 0;
int mini_batch_rotate = 0;
static bool ImagePostLoad(Image& image,  ImageLabelParseInfo& info) {

	//info.actual_w = GetNetwork().input.GetWidth();
	//info.actual_h = GetNetwork().input.GetHeight();
	info.orig_w = image.GetWidth();
	info.orig_h = image.GetHeight();
	info.crop_l = info.crop_t = 0; 
	float jitter = GetAppConfig().GetJitter();
	if (jitter > 0.0 && jitter < 0.5) {
		if (info.orig_w != info.actual_w) {
			int jitter_w = (int)(info.orig_w * jitter);
			info.crop_l = (int)rand_uniform_strong(0, jitter_w);
			info.crop_r = info.orig_w - rand_uniform_strong(0, jitter_w);
		}
		if (info.orig_h != info.actual_h) {
			int jitter_h = (int)(info.orig_h  * jitter);
			info.crop_t = (int)rand_uniform_strong(0, jitter_h);
			info.crop_b = info.orig_h - rand_uniform_strong(0, jitter_h);
		}
	}
	else {
		info.crop_r = info.orig_w;
		info.crop_b = info.orig_h;
	}
 
	//post_load_rot[rot_index] = random_gen() % ROTATE_TYPE_COUNT;
	//info.rotate =  (RotateType)(random_gen() % ROTATE_TYPE_COUNT);
	//if (info.rotate == ToLeft || info.rotate == ToRight)
		info.rotate = NotRotate;
	//旋转这块有点问题，暂时不旋转了 @2020.10.22
	//cout << post_load_rot[rot_index];
 
	//if (++rot_index >= 16) rot_index = 0;
	//else cout << ",";
	if (info.orig_h != info.actual_h || info.orig_w != info.actual_w) {
		// normally big pictures, need to shrink. Operation: to crop and then to resize		
		if (jitter > 0.0 && jitter < 0.5) {
			Image cropped;
			if (!image.Crop(cropped, info.crop_l, info.crop_t, info.crop_r - info.crop_l, info.crop_b - info.crop_t))
				return false;
			if (!cropped.ResizeTo(info.actual_w, info.actual_h, GetAppConfig().FastResize()))
				return false;
			image = cropped;
		}
		else {
			if (!image.ResizeTo(info.actual_w, info.actual_h, GetAppConfig().FastResize()))
				return false;
		}
	}

	if (!image.Rotate(info.rotate)) return false; 
	float hue = GetAppConfig().GetHue();

	info.hue = rand_uniform_strong(-hue, hue);
	info.sat = rand_scale(GetAppConfig().GetSaturation());
	info.expo = rand_scale(GetAppConfig().GetExposure()); 
	image.Distort(info.hue, info.sat, info.expo);

	return true;
}
#define RATIO_BOUND  5
static bool LoadImageLabel(const char * filename, const ImageLabelParseInfo& info) {
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Can't open label file. (This can be normal only if you use MSCOCO) \n";
		//file_error(filename);
		ofstream badfile("bad.list", ios::app);
		if (badfile.is_open()) {
			string str(filename);
			str += "\n";
			badfile.write(str.c_str(), str.length());
			badfile.close();
		}
		return false;
	}
	char line[512];
	//BoxLabelItem item; 
	info.truths->clear();
	ObjectInfo ti; 
	
	//double factor_w = 1.0 / info.orig_w;
	//double factor_h = 1.0 / info.orig_h;
	int crop_w = info.crop_r - info.crop_l;
	int crop_h = info.crop_b - info.crop_t;
	double resize_w = 1.0 / crop_w;
	double resize_h = 1.0 / crop_h; 


	float min_w = 1.0f / info.actual_w;
	float min_h = 1.0f / info.actual_h;
	int i = 0;
	double x, y, w, h;
	while (!file.eof()) {
		i++;
		file.getline(line, 512);
		if (0 == *line) continue;
		const char* line_str = line;
		ti.class_id = get_next_int(line_str);
		if (ti.class_id >= info.classes) {
			cerr << "abnormal line data in " << filename << "(" << i << ")! class_id is " << ti.class_id << "!" << endl;
			continue;
		}
		x = get_next_float(line_str);
		y = get_next_float(line_str);
		w = get_next_float(line_str);
		if (0 == *line_str) {
			cerr << "abnormal line data in " << filename << "(" << i << ")!" << endl;
			continue;
		}
		h = get_next_float(line_str);
		float orig_ratio  = w / h; 
		bool clipped = false;
		if (crop_w != info.orig_w) { 
			clipped = true;
			int left = (x - 0.5 * w) * info.orig_w - info.crop_l;
			int right = (x + 0.5 * w) * info.orig_w - info.crop_l;
			if (right < 1) continue; //裁剪掉了
			if (right >= crop_w) right = crop_w - 1;
			if (left < 0) left = 0;
			w = (right - left) * resize_w;
			x = (left + right) * resize_w * 0.5; 
 

		} 
		if (crop_h != info.orig_h) {
			clipped = true;
			int top = (y - 0.5 * h) * info.orig_h - info.crop_t;
			int bottom = (y + 0.5 * h) * info.orig_h - info.crop_t;
			if (bottom < 1) continue; //裁剪掉了
			if (bottom >= crop_h) bottom = crop_h - 1;
			if (top < 0) top = 0;
			h = (bottom - top) * resize_h;
			y = (top + bottom) * resize_h * 0.5;
		}

		if (clipped) {
			float new_ratio  = w / h;
			//裁掉3/4了，不一定能够真实反映真实的情况了，拿掉
			if (new_ratio > 4.0f * orig_ratio || orig_ratio > 4.0f * new_ratio) continue; 
		}
		switch (info.rotate) {
		case ToLeft:
			ti.w = h ;
			ti.h = w ;
			ti.x = y ;
			ti.y = 1.0f - x ;
			break;
		case ToRight:
			ti.w = h ;
			ti.h = w ;
			ti.x = 1.0f - y ;
			ti.y = x ;
			break;
		case HorizFlip:
			ti.w = w ;
			ti.h = h ;
			ti.x = 1.0f - x;
			ti.y = y ;
			break;
		case VertiFlip:
			ti.w = w ;
			ti.h = h ;
			ti.x = x ;
			ti.y = 1.0f - y ;
			break;
		case Rotate180:
			ti.w = w ;
			ti.h = h ;
			ti.x = 1.0f - x ;
			ti.y = 1.0f - y ;
			break;
		default:
			ti.w = w ;
			ti.h = h ;
			ti.x = x ;
			ti.y = y ;
			break;
		}
		
		info.truths->push_back(ti); 
	}
	file.close();
// 	if (info.truths->size() > 1) {
// 		auto seed = chrono::system_clock::now().time_since_epoch().count();
// 		shuffle(info.truths->begin(), info.truths->end(), default_random_engine(seed));
// 	}
	return true;
}

#if 0
static void load_image_in_thread_old(ImageLabelParseInfo* info) {
	Image image;
	if (!image.Load(info->file_name, info->channels)) return ;

	//long t2 = GetTickCount() ;
	//cout << " --- image file loaded in "<< (t2 - t1) << "ms. ---\n";
	if (!ImagePostLoad(image, info)) return ;

	if (!image.PullFromGPU()) return ;
	//t1 = GetTickCount();
	//cout << " --- post loaded processed in "<<(t1 - t2)<<"ms. ---\n"; 
	memcpy(info->image_data, image.GetData(), info->image_size * sizeof(float));

	//t2 = GetTickCount();
	//cout << " --- append to input in "<<(t2 - t1)<<"ms. ---\n";
	string label_path(info->file_name);

	if (!LoadImageLabel(replace_extension(label_path, ".txt"), info)) {
		cerr << "Error: Reading `" << label_path << "` failed! \n";
		return ;
	}
	//t1 = GetTickCount();
	//cout << " --- LoadImageLabel in " << (t1 - t2) << "ms. ---\n";
	if (GetAppConfig().SaveInput()) {
		char fname[MAX_PATH];
		char ext[MAX_PATH];
		const char* t = get_time_str();
		_splitpath(info->file_name, nullptr, nullptr, fname, ext);

		string str = GetAppConfig().SaveInputDir() + fname + '_' + t + ext;
		image.Save(str.c_str());
		replace_extension(str, ".txt");
		ofstream file(str);
		if (file.is_open()) {
			LPObjectInfos truths = info->truths;
			for (int i = 0;  i <(int)truths->size() ; i++) {
				ObjectInfo& truth = truths->at(i);
				file << (int)truth.class_id << " " << setprecision(6) <<
					truth.x << " " << truth.y << " " << truth.w << " " << truth.h << endl;
			}
			file.close();
		}

	}
	info->failed = false;
	//sprintf(buffer,"thread %d exit...\n", info->thread_index);
	//cout << buffer;
}

bool DataLoader::MiniBatchLoad(float* input, LPObjectInfos* truth_data , int channels, int mini_batch, 
	int width, int height, vector<string>* records) {
	
 
	//Image image; 
 
	const Dataset* ds = GetAppConfig().GetDataSet(); 
	
	int image_size = channels * width * height;

	const char* filename;
#ifdef _USE_MULTI_THREAD_
	vector<thread*> threads; 
	vector<ImageLabelParseInfo*> thread_params;
#else 
	ImageLabelParseInfo g_info;
#endif
	for (int i = 0, index = start_index; i < mini_batch; i++, index++) { 
		if (index >= ds->GetSize()) {
			filename =  ds->FilenameAt(random_gen() % ds->GetSize()).c_str();
		}
		else
			filename =  ds->FilenameAt(index).c_str();

		if (records) {
			records->push_back(filename);
		}
#ifdef _USE_MULTI_THREAD_
		ImageLabelParseInfo* info = New ImageLabelParseInfo();
#else
		ImageLabelParseInfo* info = &g_info;
#endif 
		info->small_object = GetAppConfig().SmallObjEnabled();
		info->classes = (int)ds->GetClasses();
		info->actual_w = width;
		info->actual_h = height;
		info->image_data = input;
		info->truths = truth_data[i];
		info->file_name = filename;
		info->channels = channels;
		info->image_size = image_size;
		info->failed = true;
		info->thread_index = i;
#ifdef _USE_MULTI_THREAD_
		threads.push_back(New thread(load_image_in_thread_old, info));
		thread_params.push_back(info);
#else
		load_image_in_thread(info);
		if (info->failed) {
			return false;
		}
#endif 
		input += image_size;
		
	}	
	bool succ = true;
#ifdef _USE_MULTI_THREAD_
	for(size_t i = 0; i < threads.size(); i++) {
		threads[i]->join();
		delete threads[i];
		if (thread_params[i]->failed)  succ = false;
		delete thread_params[i];
	}
#endif
	start_index += mini_batch;
	if (start_index >= ds->GetSize()) {
		start_index = 0;
		ds->ShuffleFiles();
	} 
	return succ;
}
#endif
struct ImageLoadThreadParams{
	const char* filename;
	Image* image;
	int channels;
	int cls_id; // for classifier
	bool succ;
};
static void load_image_in_thread(ImageLoadThreadParams* p) {
	p->image = New Image();
	p->succ = p->image->Load(p->filename, p->channels);
}
bool DataLoader::MiniBatchLoad(float* input, LPObjectInfos* truth_data, int channels, int mini_batch,
	int width, int height, int classes, vector<string>* records, RotateType* rotate_infos) {
	const Dataset* ds = GetAppConfig().GetDataSet();
	if (!ds) return false;
	int image_size = channels * width * height;
 
	const char* filename;
	vector<ImageLoadThreadParams*> params;
	ImageLabelParseInfo info; 
	info.classes = classes;
	info.actual_w = width;
	info.actual_h = height;
	//info.image_data = input;	
	//info.file_name = filename;
	info.channels = channels;
	info.image_size = image_size;
#ifdef _USE_MULTI_THREAD_
	vector<thread*> threads; 
#endif  
	for (int i = 0, index = start_index; i < mini_batch; i++, index++) {
		if (index >= ds->GetSize()) {
			index = random_gen() % ds->GetSize();
			filename = ds->FilenameAt(index).c_str();
		}
		else
			filename = ds->FilenameAt(index).c_str(); 
		if (records) {
			records->push_back(filename);
		}
		ImageLoadThreadParams* p = New ImageLoadThreadParams();
		*p = { filename, nullptr, channels , ds->FileClassAt(index), false };
		params.push_back(p);
#ifdef _USE_MULTI_THREAD_ 
		threads.push_back(New thread(load_image_in_thread, p));
#endif
		
	}
	start_index += mini_batch;
	if (start_index >= ds->GetSize()) {
		ds->ShuffleFiles();
		start_index = 0;
	}
	bool succ = true;
#ifdef _USE_MULTI_THREAD_
	for (int i = 0; i < (int)params.size(); i++) {
		threads[i]->join();
		delete threads[i];
		if (succ) succ = params[i]->succ;
		if (!succ) {
			delete params[i]->image;
			delete params[i];
		}
	}
	if (!succ) return false;
#endif

	for (int i = 0; i < (int)params.size(); i++) {
#ifdef _USE_MULTI_THREAD_
		Image* image = params[i]->image;
 
#else
		Image* image = New Image();
		if (!image->Load(params[i]->filename, channels)) {
			//TODO: report ;
			cerr << "Error: Reading image `" << params[i]->filename << "` failed! \n";
			delete image; 
			return false;
		}
#endif
		
		info.truths = truth_data[i];
		if (!ImagePostLoad(*image, info)) {
			succ = false;
			cerr << "Error: Processing image `" << params[i]->filename << "` failed! \n";
			continue;
		}
		if (rotate_infos) {
			rotate_infos[i] = info.rotate;
		}
		if (!image->PullFromGPU()) {
			succ = false;
			continue;
		} 
		memcpy(input, image->GetData(), image_size * sizeof(float));
		input += image_size;  
		if (params[i]->cls_id == -1) { // Release 版本调试发现params size == 25, i == 25
			string label_path(params[i]->filename);
			if (!LoadImageLabel(replace_extension(label_path, ".txt"), info)) {
				cerr << "Error: Reading `" << label_path << "` failed! \n";
				succ = false;
				continue;
			}
		}
		else {
			truth_data[i]->clear();
			truth_data[i]->push_back({ 0,0,0,0,(float)params[i]->cls_id });
		}
		delete image;
		delete params[i];
	}
	return succ;
}