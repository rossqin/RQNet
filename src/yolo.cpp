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
	GetPrevModules(element);
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
	
	output.DataType(CUDNN_DATA_FLOAT);
	Resize(input_width, input_width);
	output_channels = input_channels; 

}
YoloModule::~YoloModule() { 
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
	//TODO:����TP FP TN FN 
	DetectionResult dr;
	int classes = network->GetClassCount(); 
	results.clear();
	for (int y = 0; y < output_height; y++) {
		for (int x = 0; x < output_width; x++) {
			for (int a = 0; a < masked_anchors.size(); a++) {
				int o_offset = EntryIndex(x, y, a, INDEX_CONFIDENCE);
				dr.confidence = output_cpu[o_offset];
				if (dr.confidence < 0.5f) continue;
				// object_ness ����0.5��˵��Ԥ������Ϊ�����һ��Ŀ��
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
				Box box(dr.x, dr.y , dr.w, dr.h ); //Ԥ��� 
				bool overlaped = false;
				for (int i = 0; i < results.size(); i++) {

					if (results[i].class_id != dr.class_id) continue;
					Box box2(results[i].x, results[i].y, results[i].w, results[i].h);
					if (BoxIoU(box, box2) > GetAppConfig().NMSThreshold()) {						 
						overlaped = true;
						if (results[i].confidence < dr.confidence)   // ��ʶ����������Ŷȸ���Щ 
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
	DetectionResult dr ;
	dr.layer_index = this->layer->GetIndex();
	dr.class_id = 0;
	dr.class_confidence = 1.0f; 
	int classes = network->GetClassCount();
 
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
					exp(output_cpu[h_index]) * masked_anchors[n].height / network->GetInputHeight()) ; //Ԥ��� 
				
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
//����class��delta
void YoloModule::DeltaClass(float* output, float* delta, int cls_index, int class_id) {

	bool focal_loss = GetAppConfig().FocalLoss();
	int classes = network->GetClassCount();

	float focal_alpha = 0.5f, focal_grad = 1.0f;    // 0.25 or 0.5
	for (int c = 0; c < classes; c++, cls_index += cells_count) {
 
		float pred = output[cls_index];
		if (focal_loss) {
			if (pred > 0.0f) focal_grad = (pred - 1.0f) * (2.0f * pred * logf(pred) + pred - 1.0f);
			focal_grad *= focal_alpha;
			if (c != class_id) {
				delta[cls_index] = focal_grad * (0.0f - pred); 
			}
			else { //����ƥ���class_id
				delta[cls_index] = focal_grad * (1.0f - pred); 
			}
		}
		else {

			if (c != class_id) {
				delta[cls_index] = -pred; 
			}
			else { //����ƥ���class_id 
				delta[cls_index] = 1.0f - pred; 
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
	if (forward_input->SameShape(shortcut_delta))
		shortcut_delta = 0.0f;
	else if (!shortcut_delta.Init(network->MiniBatch(), output_channels,output_height, output_width))
		return false;

	return CalcDelta();
} 
static bool fix_precision = false;
bool YoloModule::CalcDelta(){
	int total_cells = output_height * output_width; // Ŀǰlayer�ܹ��ж��ٸ�����
	CpuPtr<float> delta_cpu(output.Elements3D()); // delta_cpu������ÿ�μ���һ���ļ���deltaʱ��deltaֵ
	CpuPtr<float> output_cpu(delta_cpu.Length()); // output_cpu��ÿ�μ���һ���ļ�ʱ��active�������ݷŵ�cpu��
  
	int offset = 0;
	int classes = network->GetClassCount(); 
	int processed_gt_count = 0;
	float d;
	perf_data = { 0 };
// 	float bg_loss = 0.0f;
// 	float obj_loss = 0.0f;
// 	float box_loss = 0.0f;
	int cfl_t_count = 0;
	float loss = 0.0f;
	for (int b = 0; b < output.Batch(); b++) { // ÿ�μ���һ���ļ�������
		
		// ��GPU��������ݸ��Ƶ�CPU
		if (!output_cpu.Pull(output.BatchData(b))) return false; 

		// ��һ������ȷ����ЩGt��Ӧ�������Ԥ���
		if(!ResolveGTs(b, cfl_t_count)) return false;
		delta_cpu.Reset();
		
		//����������һ��û�г�ͻ��GroundTruth�б�ÿ��GT����Ψһ���������

		//���������㱳���������
		if (GetAppConfig().NeedNegMining()) {
			CalcBgAnchors();
			// ���ڣ����еı���Ԥ�����bg_anchors�б����棬��һ��ƽ��������
			float balance_factor = 0.05f;//1.0f / total_cells;
			processed_gt_count += gts.size();
			float conf;
			// ���㱳����delta

			for (auto it = bg_anchors.begin(); it != bg_anchors.end(); it++) {
				int data_offset = EntryIndex(it->x, it->y, it->a, INDEX_CONFIDENCE);
				conf = output_cpu[data_offset];
				perf_data.bg_conf += conf;
				d = (0.3f - conf);
				if (d < 0.0f) {
					delta_cpu[data_offset] = balance_factor * d;
					//	bg_loss += balance_factor * d * d; 
				}

			}
		} 
		// �������и���
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
		 
		// ����Gt��Ӧ��delta
		
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

			float iou = BoxIoU(it->box, pred_box);
			perf_data.iou += iou;
			float conf = output_cpu[o_offset];
			if (conf > 0.5f)
				perf_data.gt_recall++;
			perf_data.object_conf += conf; 
			delta_cpu[o_offset] = 1.0f - conf;;
			//obj_loss += d * d;
			 
			if (classes == 1) {
				perf_data.cls_conf += 1.0f;
			}
			else {
				perf_data.cls_conf += output_cpu[o_offset + total_cells * (1 + it->class_id)];
				DeltaClass(output_cpu.ptr, delta_cpu.ptr, o_offset + total_cells, it->class_id);
			}
			float scale = (2.0f - (it->box.w * it->box.h) / total_cells);

			d = scale * (it->x_offset - output_cpu[x_offset]);
			delta_cpu[x_offset] = d;
			//box_loss += d * d; 

			d = scale * (it->y_offset - output_cpu[y_offset]);
			delta_cpu[y_offset] = d;
			//box_loss += d * d; 
			
			d = scale * (log(it->box.w / anchor_w) - output_cpu[w_offset]);
			delta_cpu[w_offset] = d;
			//box_loss += d * d; 
			
			d = scale * (log(it->box.h / anchor_h) - output_cpu[h_offset]);
			delta_cpu[h_offset] = d;
			//box_loss += d * d;
		}
		shortcut_delta.Push(delta_cpu, offset, delta_cpu.Length());
		offset += delta_cpu.Length();
		loss += square_sum_array(delta_cpu.ptr, delta_cpu.Length());
	} 
	char line[300];
	loss /= output.Batch();
// 	bg_loss /= output.Batch();
// 	obj_loss /= output.Batch();
// 	box_loss /= output.Batch();

	//float loss = bg_loss + obj_loss + box_loss;
	if (perf_data.gt_count > 0) {
		if(classes > 1)
		sprintf_s(line, 300, "  %15s: %4d/%4d recalled. conflict:%d, iou: %.6f, obj: %.4f, bg: %.4f, cls: %.4f.\n",
			name.c_str(), perf_data.gt_recall, perf_data.gt_count, cfl_t_count, perf_data.iou / perf_data.gt_count,
			perf_data.object_conf / perf_data.gt_count, perf_data.bg_conf / perf_data.bg_count,
			perf_data.cls_conf / perf_data.gt_count /*, box_loss, bg_loss, obj_loss */);
		else
			sprintf_s(line, 300, "  %15s: %4d/%4d recalled. conflict:%d, iou: %.6f, obj: %.4f.\n",
				name.c_str(), perf_data.gt_recall, perf_data.gt_count, cfl_t_count, perf_data.iou / perf_data.gt_count,
				perf_data.object_conf / perf_data.gt_count, perf_data.bg_conf / perf_data.bg_count
				/*,box_loss, bg_loss, obj_loss*/);
		std::cout << line;
	}
	/*sprintf(line, "          .5R: %.2f%%, .75R: %.2f%%,  .5P: %.2f%%, .75P: %.2f%%,.\n", 
		100.0f * perf_data.tp_50 / perf_data.gt_count, 100.0f * perf_data.tp_50 / (perf_data.tp_50 + perf_data.fp_50), 
		100.0f * perf_data.tp_75 / perf_data.gt_count,
		100.0f * perf_data.tp_75 / (perf_data.tp_75 + perf_data.fp_75));
	cout << line;*/
	network->RegisterTrainingResults(loss,perf_data.tp_50,perf_data.fp_50,perf_data.tp_75,perf_data.fp_75);
	return true;
} 
bool YoloModule::ResolveGTs(int batch,int& cfl_t_count) {
	// ��һ������ȷ����ЩGt��Ӧ�������Ԥ���
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
		// �ҵ���ƥ��������
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
		//�������GT�ǲ���Ӧ���ɱ�layer������Ԥ��
		for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {
			if (masked_anchors[a].masked_index == best_anchor_index) {
				TruthInLayer* truth = new TruthInLayer;
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
				perf_data.gt_count ++;
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
				if (NULL == candidates.ptr[c_index]) { // ��Ӧ�������û�ж�Ӧ��GT���ܺ�
					candidates.ptr[c_index] = truth;
				}
				else {
					// ��Ӧ������򼺾���GT�ˣ���һ���ĸ�IoU���
					TruthInLayer* old_t = candidates.ptr[c_index];
					if (old_t->best_iou < truth->best_iou) {
						candidates.ptr[c_index] = truth;
						truth = old_t;
					}
					// IoU�ϵ͵����ݴ���conflict_truths�����������
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

	// ����һ���������������������ϵ�GT�����
	// ����������һ��conflict_truths ���飬�����ԱΪָ��һ����ʱû�취���õ�Gt
	// ��������������Ӧ�÷�������
	// 
	cfl_t_count += conflict_truths.size();	 
	int last_confict_t = conflict_truths.size() - 1;
	while (last_confict_t >= 0) {
		// ȡ���������һ��Ԫ��
		// �п���ֱ���ҵ�һ����λ�÷�����Ҳ����˵���ڶ�best iou��Ӧ�������û��Ԥ������
		TruthInLayer* c_t = conflict_truths.at(last_confict_t);
		float conflict_iou = c_t->best_iou;
		int best_index = -1;
		float best_iou = 0.0f;
		Box object_box(0, 0, c_t->box.w, c_t->box.h);
		// �ҵ���Ŀǰ���best_iou ����һ���iou
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
		//�ҵ���û�У�
		if (best_index >= 0) { //�ҵ���
			int c_index = c_t->cell_y * output_width *  masked_anchors.size() + c_t->cell_x * masked_anchors.size() + best_index;
			if (NULL == candidates.ptr[c_index]) { // ��Ӧλ�ÿ��ţ�ֱ�Ӳ���ͺ���
				candidates.ptr[c_index] = c_t; 
				conflict_truths[last_confict_t] = NULL; 
				last_confict_t--;
			}
			else {
				//����֮ǰ�Ǹ��ǲ���iou�����ڻ���Щ
				TruthInLayer* o_t = candidates.ptr[c_index];
				if (o_t->best_iou < c_t->best_iou) { // ֮ǰ��û�����ڵĺã��µĽ�ȥ���ɵĳ���
					candidates.ptr[c_index] = c_t;
					conflict_truths[last_confict_t] =  o_t;
				}
				/*else ��������һ��*/
			}
		}
		else {
			//û�ҵ���ʵ��ûAnchor�������Gt��������ͣһ�°�
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
void YoloModule::CalcBgAnchors() {
	// ������ЩԤ����Ǳ�����
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
	perf_data.bg_count += bg_anchors.size();
}
bool YoloModule::Backward(CudaTensor& delta) {
	delta = shortcut_delta; 
	return shortcut_delta.Release();
}

bool YoloModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const {
	if (!InferenceModule::OutputIRModel(xml, bin, edges, bin_offset, l_index)) return false;
	xml << "    <layer id=\"" << index << "\" name=\"" << name << "\" precision=\"" << Precision() << "\" type=\"RegionYolo\">" << endl;
	//<data />
	//<data axis="1" classes="1" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="9"/>
	string str;
	network->GetAnchorsStr(str);
	int classes = network->GetClassCount();
	xml << "      <data anchors=\""<< str<< "\" axis=\"1\" classes=\"" << classes << "\" coords=\"4\"  do_softmax=\"0\" end_axis=\"3\" mask=\"" <<  
		mask_anchor_str <<"\" num=\""<< network->GetAnchorCount() <<"\" />" << endl;
	WritePorts(xml);
	xml << "    </layer>" << endl;
	return true;
}
uint32_t YoloModule::GetFlops() const {
	return 0;
}