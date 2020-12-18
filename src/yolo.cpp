#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "yolo.h"
#include "box.h"
#include "config.h"
#include <memory>
YoloModule::YoloModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev): 
	InferenceModule(element, l,net, prev) {
	train_bg = true;
	ParsePrevModules(element);
	const char* s = element->Attribute("anchor-masks");
	if (s)
		mask_anchor_str = s;
	else
		mask_anchor_str = ""; 
	vector<string> strs;
	split_string(strs, mask_anchor_str);
	AnchorBoxItem abi;
	for (string& s : strs) {
		abi.masked_index = atoi(s.c_str());  
		if (network->GetAnchor(abi.masked_index, abi.width, abi.height,false)) {
			masked_anchors.push_back(abi);
		}
		
	}
	bg_normalizer = element->FloatAttribute("bg-normalizer", 0.05f);
	cls_normalizer = element->FloatAttribute("class-normalizer", 1.0f);
	iou_normalizer = element->FloatAttribute("iou-normalizer", 0.25f);
	obj_normalizer = element->FloatAttribute("object-normalizer", 1.0f);
	ignore_thresh = element->FloatAttribute("ignore-thresh", 0.5f);
	truth_thresh = element->FloatAttribute("truth-thresh", 1.0f);
	objectness_smooth = element->BoolAttribute("objectness-smooth", false);
	nms_beta = element->FloatAttribute("nms-beta", 0.6f);
	max_delta =  element->FloatAttribute("max-delta", FLT_MAX); 
	output.DataType(CUDNN_DATA_FLOAT);
	Resize(input_width, input_height);
	output_channels = input_channels;  

} 
enum {
	INDEX_PROBILITY_X = 0,
	INDEX_PROBILITY_Y,  // 1
	INDEX_PROBILITY_W,  // 2
	INDEX_PROBILITY_H,  // 3
	INDEX_CONFIDENCE,   // 4
	INDEX_PROBILITY_CLASS_0,
	INDEX_PROBILITY_CLASS_1,
	INDEX_PROBILITY_CLASS_2,
	INDEX_PROBILITY_CLASS_3
	//...
};
 
int YoloModule::EntryIndex(int x, int y, int a, int channel) {
	int classes = network->GetClassCount();
	int channels = (classes == 1) ? 5 : (5 + classes); 
	return (a * channels + channel) * output.Elements2D() + y * output_width + x;
}
 
void YoloModule::UpdateDetectPerfs(float* output_cpu, vector<DetectionResult>& results) {
	//TODO:计算TP FP TN FN 
	DetectionResult dr;
	int classes = network->GetClassCount(); 
	results.clear();
	for (int y = 0; y < output_height; y++) {
		for (int x = 0; x < output_width; x++) {
			for (int a = 0; a < masked_anchors.size(); a++) {
				int o_offset = EntryIndex(x, y, a, INDEX_CONFIDENCE);
				dr.confidence = output_cpu[o_offset];
				if (dr.confidence < 0.5f) continue;
				// object_ness 大于0.5，说明预测器认为这个是一个目标
				int x_offset = EntryIndex(x, y, a, INDEX_PROBILITY_X);
				int y_offset = EntryIndex(x, y, a, INDEX_PROBILITY_Y);
				int w_offset = EntryIndex(x, y, a, INDEX_PROBILITY_W);
				int h_offset = EntryIndex(x, y, a, INDEX_PROBILITY_H);
				float anchor_w = masked_anchors[a].width  / network->GetInputWidth() * output_width;
				float anchor_h = masked_anchors[a].height / network->GetInputHeight() * output_height;
				dr.class_id = -1;
				dr.class_confidence = 0.0f;
				if (classes > 1) {
					for (int off = o_offset + output.Elements2D(), c = 0; c < classes; c++, off += output.Elements2D()) {
						if (output_cpu[off] > dr.class_confidence) {
							dr.class_confidence = output_cpu[off]  ;
							dr.class_id = c;
						}
					}
				}
				else {
					dr.class_confidence = 1.0f;
					dr.class_id = 0;
				}
				dr.x = x + output_cpu[x_offset];
				dr.y = y + output_cpu[y_offset];
				dr.w = exp(output_cpu[w_offset]) * anchor_w;
				dr.h = exp(output_cpu[h_offset]) * anchor_h;
				Box box(dr.x, dr.y , dr.w, dr.h ); //预测框 
				bool overlaped = false;
				for (int i = 0; i < results.size(); i++) {

					if (results[i].class_id != dr.class_id) continue;
					Box box2(results[i].x, results[i].y, results[i].w, results[i].h);
					if (BoxIoU(box, box2) > GetAppConfig().NMSThreshold()) {						 
						overlaped = true;
						if (results[i].confidence < dr.confidence)   // 新识别的物体置信度更高些 
							results[i] = dr;
					 
						break;
					}
				}
				if (!overlaped) {
					results.push_back(dr);
				}

			}
		}
	}
}

bool YoloModule::Resize(int w, int h) {
	input_height = h;
	input_width = w;
	output_width = input_width;
	output_height = input_height; 
	cells_count = output_height * output_width;
	return true;
}

bool YoloModule::Detect() {
	//std::cout << "   Start Analyze the " << input_width << "x" << input_height << " layer ...\n";
	CpuPtr<float> output_cpu(output.Elements3D(), output.Data());
	if (output_cpu.GetError()) {
		cerr << " Error: Copy memory failed in " << name << "!\n";
		return false;
	}
	int classes = network->GetClassCount(); 

	DetectionResult dr ;
	dr.layer_index = this->layer->GetIndex();
	dr.class_id = 0;
	dr.class_confidence = 1.0f; 
	
 
	float threshold = GetAppConfig().ThreshHold();
	for (int y = 0,i = 0; y < output.Height(); y++) {
		for (int x = 0; x < output.Width(); x++, i++) { 
			for (int n = 0; n < (int)masked_anchors.size(); n++) {
				int x_index = EntryIndex(x,y, n, INDEX_PROBILITY_X); 
				int y_index = x_index + cells_count;
				int w_index = y_index + cells_count;
				int h_index = w_index + cells_count;
				int conf_index = h_index + cells_count; 
				dr.confidence = output_cpu[conf_index];
				if (dr.confidence < threshold) {
					continue;
				}
				if (classes > 1) {
					int class_id = -1;
					float best_cls_conf = 0.0f;
					int cls_index = conf_index + cells_count;
					for (int c = 0; c < classes; c++, cls_index += cells_count) {
						if (output_cpu[cls_index] > threshold &&
							output_cpu[cls_index] > best_cls_conf) {
							class_id = c;
							best_cls_conf = output_cpu[cls_index];
						}
					}
					if (-1 == class_id) {
						cout << "Class confidence below threshold, ignore...\n";
						continue;
					}
					dr.class_id = class_id;
					dr.class_confidence = dr.confidence * best_cls_conf;
				} 
				Box box((x + output_cpu[x_index]) / input_width,
					(y + output_cpu[y_index]) / input_height,
					exp(output_cpu[w_index]) * masked_anchors[n].width / network->GetInputWidth(),
					exp(output_cpu[h_index]) * masked_anchors[n].height / network->GetInputHeight()) ; //预测框 
				
				dr.x = box.x;
				dr.y = box.y;
				dr.w = box.w;
				dr.h = box.h; 
				network->AddDetectionResult(dr);
			} 
		}
	}
	return true;
}
//计算class的delta
void YoloModule::DeltaClass(float* o, float* d, int cls_index, int class_id) {

	bool focal_loss = GetAppConfig().FocalLoss();
	int classes = network->GetClassCount();

	float focal_alpha = 0.5f, focal_grad = 1.0f;    // 0.25 or 0.5
	for (int c = 0; c < classes; c++, cls_index += cells_count) {
 
		float pred = o[cls_index];
		if (focal_loss) {
			if (pred > 0.0f) focal_grad = (pred - 1.0f) * (2.0f * pred * logf(pred) + pred - 1.0f);
			focal_grad *= focal_alpha;
			if (c != class_id) {
				d[cls_index] = focal_grad * (0.0f - pred); 
			}
			else { //这是匹配的class_id
				d[cls_index] = focal_grad * (1.0f - pred); 
			}
		}
		else {

			if (c != class_id) {
				d[cls_index] = -pred; 
			}
			else { //这是匹配的class_id 
				d[cls_index] = 1.0f - pred; 
			}

		} 
	}
}
 
bool YoloModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false;
	CudaTensor* forward_input = context.input ? context.input : &input;
	if (forward_input->DataFormat() != CUDNN_TENSOR_NCHW) return false;
	int b, n;
	int classes = network->GetClassCount();
	int expected_channels;
	if(classes > 1)
		expected_channels = (classes + 5) * masked_anchors.size();
	else
		expected_channels = 5 * masked_anchors.size(); 

	if (input_channels != expected_channels) {
		cerr << " Error: Input channel count should be " << expected_channels << "! \n";
		return false;
	}


	if (forward_input->DataType() == CUDNN_DATA_FLOAT) {
		output = *(forward_input);
	}
	else {
		float *out = reinterpret_cast<float *>(output.Data());
		half *in = reinterpret_cast<__half*>(forward_input->Data());
		if(!f16_to_f32( out, in, forward_input->Elements())) return false;
	}
		
	float* output_gpu = reinterpret_cast<float*>(output.Data()); 
	bool ret = true; 
	int total_cells = output_height * output_width;
	int xy_elements = 2 * total_cells;
	
	int conf_cls_elements = total_cells;
	if (classes > 1) {
		conf_cls_elements = (1 + classes) * total_cells;
	}
	for (b = 0; b < output.Batch(); b++) {
		for (n = 0; n < (int)masked_anchors.size(); n++) {
			float* x = output_gpu + EntryIndex(0,0,n, INDEX_PROBILITY_X);
			if (!activate_array_ongpu(x,x, xy_elements, output.DataType(), LOGISTIC)) {
				return false;
			} 
			x += (INDEX_CONFIDENCE - INDEX_PROBILITY_X) * total_cells;
			if (!activate_array_ongpu(x, x, conf_cls_elements, output.DataType(), LOGISTIC))
				return false; 
		}
		output_gpu += total_cells * output_channels;
	} 
  	if (!context.training) {   
  		return Detect();
  	}
	if (forward_input->Like(shortcut_delta))
		shortcut_delta = 0.0f;
	else if (!shortcut_delta.Init({ network->MiniBatch(), output_channels,output_height, output_width }))
		return false;

	return CalcDelta();
} 
static inline void fix_nan_inf(float& val) {
	if (isnan(val) || isinf(val)) {
		val = 0.0f;
	}
}

static inline void clip_value(float& val, const float max_val) {
	if (val > max_val) { 
		val = max_val;
	}
	else if (val < -max_val) { 
		val = -max_val;
	} 
}

float YoloModule::DeltaBox(float* o, float* d, const TruthInLayer& gt, Box pred, float anchor_w, float anchor_h) {
	 
	if (pred.w == 0) { pred.w = 1.0f; }
	if (pred.h == 0) { pred.h = 1.0f; }

	float loss = 0.0f;

	int total_cells = output_width * output_height;
	if (GetAppConfig().UseCIoULoss())    {  // use ciou
		
		float delta_x = 0.0f, delta_y = 0.0f, delta_w = 0.0f, delta_h = 0.0f;

		loss = DeltaCIoU(pred, gt.box, delta_x, delta_y, delta_w, delta_h);
		fix_nan_inf(delta_x);
		fix_nan_inf(delta_y);
		fix_nan_inf(delta_w);
		fix_nan_inf(delta_h);

		if (max_delta != FLT_MAX) {
			clip_value(delta_x, max_delta);
			clip_value(delta_y, max_delta);
			clip_value(delta_w, max_delta);
			clip_value(delta_h, max_delta);
		}
		d[0] += delta_x * iou_normalizer;
		d[total_cells] += delta_y * iou_normalizer;
		d[total_cells * 2] += delta_w * iou_normalizer;
		d[total_cells * 3] += delta_h * iou_normalizer;
	}
	else {
		float scale = (2.0f - (gt.box.w * gt.box.h) / total_cells) * iou_normalizer;

		float delta_x = (gt.x_offset - o[0]);
		float delta_y = (gt.y_offset - o[total_cells]);
		float delta_w = (log(gt.box.w / anchor_w) - o[total_cells * 2]);
		float delta_h = (log(gt.box.h / anchor_h) - o[total_cells * 3]);

		d[0] = scale * delta_x;
		d[total_cells] = scale * delta_y;
		d[total_cells * 2] = scale * delta_w;
		d[total_cells * 3] = scale * delta_h;

		loss = 0.5f * (delta_x * delta_x + delta_y * delta_y + delta_w * delta_w + delta_h * delta_h);


	}
	return loss;
	 
}
const float OBJECTNESS_TARGET = 0.7f;

bool YoloModule::CalcDelta(){
	int total_cells = output_height * output_width; // 目前layer总共有多少个格子
	CpuPtr<float> delta_cpu(output.Elements3D()); // delta_cpu就是在每次计算一个文件的delta时的delta值
	CpuPtr<float> output_cpu(delta_cpu.Length()); // output_cpu是每次计算一个文件时把active过的数据放到cpu内
  
	int offset = 0;
	int classes = network->GetClassCount(); 
	int processed_gt_count = 0;
	float d;
	TrainingResult result = {  0 }; 

	int cfl_t_count = 0; 
	for (int b = 0; b < output.Batch(); b++) { // 每次计算一个文件的数据
		
		// 把GPU里面的数据复制到CPU
		if (!output_cpu.Pull(output.BatchData(b))) return false; 

		// 第一步：先确定哪些Gt是应该在这层预测的
		if(!ResolveGTs(b, cfl_t_count,result.gt_count)) return false;
		delta_cpu.Reset();
		
		//现在我们有一个没有冲突的GroundTruth列表，每个GT都有唯一的先验框负责

		//接下来计算背景的先验框
		if (GetAppConfig().NeedNegMining()) {
			CalcBgAnchors(result.bg_count);
			// 现在，所有的背景预测框在bg_anchors列表里面，用一个平衡因子来
			 
			processed_gt_count += gts.size();
			float conf;
			// 计算背景的delta

			for (auto it = bg_anchors.begin(); it != bg_anchors.end(); it++) {
				int data_offset = EntryIndex(it->x, it->y, it->a, INDEX_CONFIDENCE);
				conf = output_cpu[data_offset];
				result.bg_conf += conf;
				d = (0.3f - conf);
				if (d < 0.0f) {
					delta_cpu[data_offset] = bg_normalizer * d; 
				}

			}
		} 
		// 降低误判概率
		for (int y = 0; y < output_height; y++) {
			for (int x = 0; x < output_width; x++) {
				for (int a = 0; a < masked_anchors.size(); a++) {
					float anchor_w, anchor_h;
					network->GetAnchor(masked_anchors.at(a).masked_index, anchor_w, anchor_h);
					int x_index = EntryIndex(x, y, a, INDEX_PROBILITY_X);
					Box pred_box(x + output_cpu.ptr[x_index],
						y + output_cpu.ptr[x_index + total_cells],
						anchor_w * output_width * exp(output_cpu.ptr[x_index + 2 * total_cells]),
						anchor_h * output_height * exp(output_cpu.ptr[x_index + 3 * total_cells]));
					float best_iou = 0;
					for (auto gt : gts) {
						float iou = BoxIoU(pred_box, gt.box);
						if (iou > best_iou) {
							best_iou = iou;
						}
					}
					int o_index = x_index + 4 * total_cells;
					delta_cpu.ptr[o_index] = 0.0f;
					if (best_iou < 0.7f) {
						if (output_cpu.ptr[o_index] > 0.3f)
							delta_cpu.ptr[o_index] = 0.3f - output_cpu.ptr[o_index];
					}
				}
			}
		}
		 
		// 计算Gt对应的delta
		
		for (auto it = gts.begin(); it != gts.end(); it++) {
			int x_offset = EntryIndex(it->cell_x, it->cell_y, it->best_anchor, INDEX_PROBILITY_X);
			int y_offset = x_offset + cells_count;
			int w_offset = x_offset + cells_count * 2;
			int h_offset = x_offset + cells_count * 3;
			int o_offset = x_offset + cells_count * 4;
			float anchor_w = masked_anchors[it->best_anchor].width / network->GetInputWidth() * output_width;
			float anchor_h = masked_anchors[it->best_anchor].height / network->GetInputHeight() * output_height;
			Box pred_box(it->cell_x + output_cpu[x_offset], it->cell_y + output_cpu[y_offset],
				exp(output_cpu[w_offset]) * anchor_w,
				exp(output_cpu[h_offset]) * anchor_h);

			float conf = output_cpu[o_offset];
			if (conf > GetAppConfig().ThreshHold())
				result.gt_recall++;
			result.object_conf += conf; 
			if (GetAppConfig().FocalLoss()) {
				delta_cpu[o_offset] = focal_loss_delta(conf,0.5f,2.0f) * obj_normalizer;
			}
			else{
				if (conf < OBJECTNESS_TARGET)
					delta_cpu[o_offset] = (OBJECTNESS_TARGET - conf) * obj_normalizer;
				else
					delta_cpu[o_offset] = 0.0f;
			}
			 
			if (classes == 1) {
				result.cls_conf += 1.0f;
			}
			else {
				result.cls_conf += output_cpu[o_offset + total_cells * (1 + it->class_id)];
				DeltaClass(output_cpu.ptr, delta_cpu.ptr, o_offset + total_cells, it->class_id);
			}
			float iou = BoxIoU(it->box, pred_box);
			result.iou += iou;
			if (iou > 0.5f) {
				result.recalls++;
				if(iou > 0.75f)
					result.recalls_75++;
			}  
			result.ciou += BoxCIoU(it->box, pred_box); 

			result.box_loss += DeltaBox(output_cpu.ptr + x_offset, delta_cpu.ptr + x_offset, *it, pred_box, anchor_w, anchor_h);

		}
		shortcut_delta.Push(delta_cpu, offset, delta_cpu.Length());
		offset += delta_cpu.Length(); 
		for (int a = 0; a < masked_anchors.size(); a++) {
			float* p = delta_cpu.ptr;
			if (classes == 1)
				p += a * 5 * total_cells;
			else
				p += a * (5 + classes) * total_cells;
			float obj_loss = square_sum_array(p + 4 * total_cells, total_cells) / (obj_normalizer * obj_normalizer);
			result.obj_loss += obj_loss; 
			if (classes > 1) {
				result.cls_loss += square_sum_array(p + 5 * total_cells, total_cells * classes);
			}
		}
		result.old_loss += square_sum_array(delta_cpu.ptr, delta_cpu.Length());
		
	} 
	result.old_loss /= output.Batch();
	result.obj_loss /= output.Batch();
	result.cls_loss /= output.Batch();
	result.box_loss /= output.Batch();
	
	stringstream ss;
	ss << "    " << setw(20) << right << name << ":" << setw(4) << result.gt_recall << "/" << setw(4) << left << result.gt_count << " recalled. ";

	if (result.gt_count > 0) {
		ss << fixed << setprecision(4) << " obj: " << result.object_conf / result.gt_count;
		if (result.bg_count > 0) {
			ss << ", bg: " << result.bg_conf / result.bg_count;
		}
		if (classes > 1)
			ss << ", class conf: " << result.cls_conf / result.gt_count ;
		ss << ", iou: " <<  result.iou / result.gt_count ; 
	}

	cout << ss.str() << endl;
	network->AddTrainingResult(result);
	return true;
} 
bool YoloModule::ResolveGTs(int batch,int& cfl_t_count,int& gt_count) {
	// 第一步：先确定哪些Gt是应该在这层预测的
	const LPObjectInfos all_truths = network->GetBatchTruths(batch);
	if (!all_truths) return false;
	vector<TruthInLayer*> conflict_truths;
	CpuPtr<TruthInLayer*> candidates(output_width * output_height * masked_anchors.size()); 
	//CpuPtr<int> mappings(output_width * output_height * masked_anchors.size());
	Box anchor_box(0.0f, 0.0f, 0.0f, 0.0f);
	int truths_in_layer = 0;
	gts.clear();
	for (int t = 0; t < (int)all_truths->size(); t++) {
		const ObjectInfo& object = all_truths->at(t);
		Box object_box(0, 0, object.w, object.h);
		int best_anchor_index = -1;
		float best_iou = 0.0f;
		float best_anchor_width = 0.0f, best_anchor_height = 0.0f;
		// 找到最匹配的先验框
		for (int a = network->GetAnchorCount() - 1; a >= 0; a--) {
			network->GetAnchor(a, anchor_box.w, anchor_box.h);
			float iou = BoxIoU(object_box, anchor_box);
			if (iou > best_iou) {
				best_anchor_index = a;
				best_iou = iou;
				best_anchor_width = anchor_box.w * output_width;
				best_anchor_height = anchor_box.h * output_height;
			}
		}
		//看看这个GT是不是应该由本layer来负责预测
		for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {
			if (masked_anchors[a].masked_index == best_anchor_index) {
				TruthInLayer* truth = New TruthInLayer;
				memset(truth, 0, sizeof(TruthInLayer));
				truth->box = { object.x * output_width, object.y * output_height,
						object.w * output_width, object.h * output_height };
				truth->cell_x = floor(truth->box.x);
				 
				truth->x_offset = truth->box.x - truth->cell_x;
				truth->cell_y = floor(truth->box.y);
				truth->y_offset = truth->box.y - truth->cell_y;
				truth->orig_index = t; 

				truth->best_anchor = a;
				truth->best_iou = best_iou;
				truth->class_id = object.class_id;
				/*cout << "\t (" << object.x << ", " << object.y << ", " << object.w << ", " << object.h
					<< ") matched to " << name.c_str() << "( " << truth->cell_x << ", " <<
					truth->cell_y << ", " << a << " ) with max_iou " << best_iou << endl;
				//*/
				gt_count ++;
				int c_index = (truth->cell_y * output_width  + truth->cell_x )
					* masked_anchors.size() + a;
				truths_in_layer++; 
#if 0
				mappings.ptr[c_index] = mappings.ptr[c_index] + 1;
				if (mappings.ptr[c_index] > 1) {
					cout << "\t  *** duplicated truths predictor by same anchor :( " << truth->cell_x << ", " << truth->cell_y << ", "
						<< a << ")\r\n";
				}
				gts.push_back(*truth);
				delete truth;
#else 
				if (NULL == candidates.ptr[c_index]) { // 相应的先验框还没有对应的GT，很好
					candidates.ptr[c_index] = truth;
				}
				else {
					// 相应的先验框己经有GT了，看一下哪个IoU最高
					TruthInLayer* old_t = candidates.ptr[c_index];
					if (old_t->best_iou < truth->best_iou) {
						candidates.ptr[c_index] = truth;
						truth = old_t;
					}
					// IoU较低的先暂存在conflict_truths里面后续处理
					conflict_truths.push_back(truth);
				}
#endif
				break;
			}
		}
	}
#if 0
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	shuffle(gts.begin(), gts.end(), default_random_engine(seed));

#else

	// 处理一个先验框里面出现两个以上的GT的情况
	// 现在我们有一个conflict_truths 数组，数组成员为指向一个暂时没办法安置的Gt
	// 下面来挨个计算应该放在哪里
	// 
	cfl_t_count += conflict_truths.size();	 
	int last_confict_t = conflict_truths.size() - 1;
	while (last_confict_t >= 0) {
		// 取出数组最后一个元素
		// 有可能直接找到一个新位置放它，也就是说，第二best iou对应的先验框还没有预测任务
		TruthInLayer* c_t = conflict_truths.at(last_confict_t);
		float conflict_iou = c_t->best_iou;
		int best_index = -1;
		float best_iou = 0.0f;
		Box object_box(0, 0, c_t->box.w, c_t->box.h);
		// 找到比目前这个best_iou 还次一点的iou
		for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {			 
			network->GetAnchor(masked_anchors.at(a).masked_index, anchor_box.w, anchor_box.h);
			anchor_box.w *= output_width;
			anchor_box.h *= output_height;
			float iou = BoxIoU(object_box, anchor_box);
			if (iou >= conflict_iou) continue;
			if (iou > best_iou) {
				best_index = a;
				best_iou = iou;
				c_t->best_anchor = a;
				c_t->best_iou = iou;
			}
		}
		//找到了没有？
		if (best_index >= 0) { //找到了
			int c_index = c_t->cell_y * output_width *  masked_anchors.size() + c_t->cell_x * masked_anchors.size() + best_index;
			if (NULL == candidates.ptr[c_index]) { // 相应位置空着，直接插入就好了
				candidates.ptr[c_index] = c_t; 
				conflict_truths[last_confict_t] = NULL; 
				last_confict_t--;
			}
			else {
				//看看之前那个是不是iou比现在还好些
				TruthInLayer* o_t = candidates.ptr[c_index];
				if (o_t->best_iou < c_t->best_iou) { // 之前的没有现在的好，新的进去，旧的出来
					candidates.ptr[c_index] = c_t;
					conflict_truths[last_confict_t] =  o_t;
				}
				/*else 继续找下一个*/
			}
		}
		else {
			//没找到，实在没Anchor安放这个Gt，报个错，停一下吧
			cerr << "\t\tUnexcepcted situation: more than 1 groundtruths predicted by anchor box( x:"
				<< c_t->cell_x << ",y:" << c_t->cell_y << ",masked:" << c_t->best_anchor << "): \r\n";
			cerr << "\t\t -- File: " << network->current_training_files.at(batch).c_str() << "(" << c_t->orig_index << "), Layer:" << this->name.c_str() << ".\r\n\t\t !!! Failed to reschedule.\r\n\r\n";

			conflict_truths[last_confict_t] =  NULL;
			last_confict_t--;
			truths_in_layer--;
			delete c_t;
			// system("pause");
		}
	}
	gts.clear();
	for (int i = 0; i < candidates.Length(); i++) {
		TruthInLayer* truth = candidates.ptr[i];
		if(!truth) continue;
		gts.push_back(*truth);
		delete truth;
	}
#endif
	return true;
}
void YoloModule::CalcBgAnchors(int& bg_count) {
	// 看看哪些预测框是背景框
	bg_anchors.clear();
	for (int y = 0; y < output_height; y++) {
		for (int x = 0; x < output_width; x++) {
			for (int a = 0; a < masked_anchors.size(); a++) {
				bool bg = true;
				Box a_box(0, 0, masked_anchors[a].width / network->GetInputWidth() * output_width,
					masked_anchors[a].height / network->GetInputHeight() * output_height);
				for (auto it = gts.begin(); it != gts.end(); it++) {
					if (it->cell_x == x && it->cell_y == y && it->best_anchor == a) {
						bg = false; 
						break;
					}
					if (x > it->cell_x)
						a_box.x = x;
					else if (x == it->cell_x)
						a_box.x = it->box.x;
					else
						a_box.x = x + 1;
					if (y > it->cell_y)
						a_box.y = y;
					else if (y == it->cell_y)
						a_box.y = it->box.y;
					else
						a_box.y = y + 1;

					float iou = BoxIoU(a_box, it->box);
					if (iou > 0.0f) {
						bg = false;
						break;
					}
				}
				if (bg) {
					bg_anchors.push_back({ x,y,a });
				}
			}
		}
	}
	bg_count += bg_anchors.size();
}
bool YoloModule::Backward(CudaTensor& delta) {
	delta = shortcut_delta; 
	return shortcut_delta.Release();
} 
void YoloModule::WriteOpenVINOOutput(ofstream& xml) const {
	xml << "\t\t<output id=\"" << name << "\" base=\"" << prevs[0].module->Name() << "\" anchor-masks=\"" << mask_anchor_str << "\" />\n";
}
uint32_t YoloModule::GetFlops() const {
	return 0;
}