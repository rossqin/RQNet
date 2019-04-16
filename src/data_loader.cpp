#include "stdafx.h"
#include "network.h"
#include "config.h"
#include "data_loader.h"

int post_load_rot[16] = { 3,4,1,5,0,1,0,1,3,1,4,0,3,4,4,1 };
int rot_index = 0;
int mini_batch_rotate = 0;
bool DataLoader::ImagePostLoad(Image & image,  ImageLabelParseInfo & info) {

	//info.actual_w = GetNetwork().input.GetWidth();
	//info.actual_h = GetNetwork().input.GetHeight();
	info.orig_w = image.GetWidth();
	info.orig_h = image.GetHeight();
	info.crop_l = info.crop_t = 0; 
	float jitter = GetAppConfig().GetJitter();
	if (jitter > 0.0 && jitter < 0.5) {
		if (info.orig_w != info.actual_w) {
			int dw = (int)(info.orig_w * jitter);
			info.crop_l = (int)rand_uniform_strong(0, dw);
			info.crop_r = info.orig_w - rand_uniform_strong(0, dw);
		}
		if (info.orig_h != info.actual_h) {
			int dh = (int)(info.orig_h  * jitter);
			info.crop_t = (int)rand_uniform_strong(0, dh);
			info.crop_b = info.orig_h - rand_uniform_strong(0, dh);
		}
	}
	else {
		info.crop_r = info.orig_w;
		info.crop_b = info.orig_h;
	}
 
	//post_load_rot[rot_index] = random_gen() % ROTATE_TYPE_COUNT;
	info.rotate = (RotateType)(random_gen() % ROTATE_TYPE_COUNT);
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
bool DataLoader::LoadImageLabel(const char * filename, ObjectInfo* labels, const ImageLabelParseInfo& info) {
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
	vector<ObjectInfo> items;
	ObjectInfo ti;
	float lowest_w, lowest_h, factor_w, factor_h;
	int crop_w, crop_h;
	//
	if (info.actual_w) {
		lowest_w = 1.0f / info.actual_w;
		lowest_h = 1.0f / info.actual_h;

		crop_w = info.crop_r - info.crop_l;
		crop_h = info.crop_b - info.crop_t;
		factor_w = 1.0f / crop_w;
		factor_h = 1.0f / crop_h;
	}

	int x, y, w, h, l, r, t, b, i = 1;
	while (!file.eof()) {
		file.getline(line, 512);
		if (0 == *line) continue;
		const char* line_str = line;
		ti.class_id = get_next_int(line_str);
		if (ti.class_id >= info.classes) {
			cerr << "abnormal line data in " << filename << "(" << i++ << ")! class_id is " << ti.class_id << "!" << endl;
			continue;
		}
		ti.x = get_next_float(line_str);
		ti.y = get_next_float(line_str);
		ti.w = get_next_float(line_str);
		if (0 == *line_str) {
			cerr << "abnormal line data in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}
		ti.h = get_next_float(line_str);

		if (info.orig_h == info.actual_h && info.orig_w == info.actual_w) {
			i++;
			items.push_back(ti);
			if (i > info.max_boxes) break;
			continue;
		}

		// original coordinates 
		x = (int)(ti.x * info.orig_w);
		y = (int)(ti.y * info.orig_h);
		w = (int)(ti.w * info.orig_w);
		h = (int)(ti.h * info.orig_h);


		l = x - (w >> 1); r = l + w; t = y - (h >> 1); b = t + h;

		// transform
		if (l >= info.crop_r) {
			//cerr << "  Box left out bound in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}

		l -= info.crop_l; if (l < 0) l = 0;

		if (r <= info.crop_l) {
			//cerr << "  Box right out bound in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}
		if (r > info.crop_r) r = info.crop_r;
		r -= info.crop_l;
		if (r <= 0) {
			//cerr << "  Box right out bound in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}

		if (t >= info.crop_b) {
			//cerr << "  Box top out bound in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}
		t -= info.crop_t;
		if (t < 0) t = 0;
		if (b > info.crop_b) b = info.crop_b;
		b -= info.crop_t;

		if (b <= 0) {
			//cerr << "  Box bottom out bound in " << filename << "(" << i++ << ")!" << endl;
			continue;
		}
		x = (l + r) >> 1;
		y = (b + t) >> 1;
		w = r - l;
		h = b - t;
		// 由于裁剪导致狭长的图片出现，这种不要

		if (l == 0 || r == info.crop_r) {
			if (h > w * RATIO_BOUND) {
				// cerr << "Box discarded because of cropping in *" << filename << "(" << i++ << ")!" << endl;
				continue;
			}
		}
		if (t == 0 || b == info.crop_b) {
			if (w > h * RATIO_BOUND) {
				// cerr << "Box discarded because of cropping in -" << filename << "(" << i++ << ")!" << endl;
				continue;
			}
		}
		if (!info.small_object) {
			if (h < 3 && w < 3) { //太小的不要了
								  //cerr << "Box discarded because it's too small in -" << filename << "(" << i++ << ")!" << endl;
				continue;
			}
		}
		switch (info.rotate) {
		case ToLeft:
			ti.w = h * factor_h;
			ti.h = w * factor_w;
			ti.x = y * factor_h;
			ti.y = 1.0f - x * factor_w;
			break;
		case ToRight:
			ti.w = h * factor_h;
			ti.h = w * factor_w;
			ti.x = 1.0f - y * factor_h;
			ti.y = x * factor_w;
			break;
		case HorizFlip:
			ti.w = w * factor_w;
			ti.h = h * factor_h;
			ti.x = 1.0f - x * factor_w;
			ti.y = y * factor_h;
			break;
		case VertiFlip:
			ti.w = w * factor_w;
			ti.h = h * factor_h;
			ti.x = x * factor_w;
			ti.y = 1.0f - y * factor_h;
			break;
		case Rotate180:
			ti.w = w * factor_w;
			ti.h = h * factor_h;
			ti.x = 1.0f - x * factor_w;
			ti.y = 1.0f - y * factor_h;
			break;
		default:
			ti.w = w * factor_w;
			ti.h = h * factor_h;
			ti.x = x * factor_w;
			ti.y = y * factor_h;
			break;
		}


		i++;
		items.push_back(ti);
		if (i > info.max_boxes) break;
	}
	file.close();
	//// end of 
	ObjectInfo * item = labels;
	i = 0; 
	while (items.size() > 0) {
		int index = random_gen() % items.size();
		*item++ = items[index];
		items.erase(items.begin() + index);
		i++;
	} 
	if (i < GetAppConfig().GetMaxTruths()) item->class_id = -1;
	return true;
}

const char* rotate_to_str(RotateType rt) {
	switch (rt)
	{
	case ToLeft:
		return "rleft";
	case ToRight:
		return "rright";
	case HorizFlip:
		return "hflip";
	case VertiFlip:
		return "vflip";
	case Rotate180:
		return "rot180";
	default:
		return "normal";
	}
}
bool DataLoader::MiniBatchLoad(float* input, ObjectInfo* truth_data , int channels, int mini_batch, int width, int height) {
	
 
	//Image image;
	ImageLabelParseInfo info = { 0 };

 
	const Dataset* ds = GetAppConfig().GetDataSet();
	info.max_boxes = GetAppConfig().GetMaxTruths();
	info.small_object = GetAppConfig().SmallObjEnabled() ;
	info.classes = (int)ds->GetClasses();
	info.actual_w = width;
	info.actual_h = height;
	
	int image_size = channels * width * height;

 
	for (int i = 0, index = start_index; i < mini_batch; i++, index++) { 
		if (index >= ds->GetSize()) {
			info.file_name =  ds->FilenameAt(random_gen() % ds->GetSize()).c_str();
		}
		else
			info.file_name =  ds->FilenameAt(index).c_str(); 
		//cout << info.file_name << endl;
		//cout << "*** loading " << info.file_name << " *** \n";
		long t1 = GetTickCount();
		Image image;
		if (!image.Load(info.file_name, channels)) return false;  
		
		long t2 = GetTickCount() ;
		//cout << " --- image file loaded in "<< (t2 - t1) << "ms. ---\n";
		if (!ImagePostLoad(image, info)) return false; 

		if (!image.PullFromGPU()) return false;
		t1 = GetTickCount();
		//cout << " --- post loaded processed in "<<(t1 - t2)<<"ms. ---\n"; 
		memcpy(input, image.GetData(), image_size * sizeof(float)); 
		input += image_size;
		t2 = GetTickCount();
		//cout << " --- append to input in "<<(t2 - t1)<<"ms. ---\n";
		string label_path(info.file_name);
		
		if (!LoadImageLabel(replace_extension(label_path, ".txt"), truth_data, info)) {
			cerr << "Error: Reading `" << label_path  << "` failed! \n";
			return false;
		}
		t1 = GetTickCount();
		//cout << " --- LoadImageLabel in " << (t1 - t2) << "ms. ---\n";
		if (GetAppConfig().SaveInput()) { 
			char fname[MAX_PATH];
			char ext[MAX_PATH]; 
			const char* t = get_time_str();
			_splitpath(info.file_name, NULL, NULL, fname, ext);			
		 
			string str = GetAppConfig().SaveInputDir() + fname + '_' + t + ext ; 
			image.Save(str.c_str());
			replace_extension(str, ".txt");
			ofstream file(str);
			if (file.is_open()) {
				ObjectInfo* temp = truth_data; 
				while (temp->class_id != -1) {
					file << (int)temp->class_id << " " << setprecision(6) <<
						temp->x << " " << temp->y << " " << temp->w << " " << temp->h << endl;					 
					temp++;
				}
				file.close();
			}

		}
		truth_data += GetAppConfig().GetMaxTruths();
	}	
	 
	start_index += mini_batch;
	if (start_index >= ds->GetSize()) {
		start_index = 0;
		ds->ShuffleFiles();
	} 
	return true;
}
