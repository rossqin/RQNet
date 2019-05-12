#include "stdafx.h"
#include "network.h"
#include "param_pool.h"
#include "yolo.h"
#include "box.h"
#include "config.h"
#include <memory>
YoloModule::YoloModule(const XMLElement* element, Layer* l, CNNNetwork* net, InferenceModule* prev): 
	InferenceModule(element, l,net, prev) {
	 
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
		if (network->GetAnchor(abi.masked_index, abi.width, abi.height)) {
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
 
int YoloModule::EntryIndex(int anchor, int loc, int entry) {
 
	return (anchor * 5 + entry) * output.Elements2D() + loc;
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
	std::cout << "Start Analyze the " << input_width << "x" << input_height << " layer ...\n";
	CpuPtr<float> output_cpu(output.Elements3D(), output.Data());
	if (output_cpu.GetError()) {
		cerr << " Error: Copy memory failed in " << name << "!\n";
		return false;
	}
	DetectionResult dr ;
	float reciprocal_w = 1.0f / input_width;
	float reciprocal_h = 1.0f / input_height;
	int classes = network->GetClassCount();
 
	float threshold = GetAppConfig().ThreshHold();
	for (int y = 0,i = 0; y < output.Height(); y++) {
		for (int x = 0; x < output.Width(); x++, i++) { 
			for (int n = 0; n < (int)masked_anchors.size(); n++) {
				int x_index = EntryIndex(n, i, INDEX_PROBILITY_X); 
				int y_index = x_index + cells_count;
				int w_index = y_index + cells_count;
				int h_index = w_index + cells_count;
				int conf_index = h_index + cells_count; 
				dr.confidence = output_cpu[conf_index];
				if (dr.confidence < threshold) {
					continue;
				}
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
					std::cout << "Class confidence below threshold, ignore...\n";
					continue;
				} 
				dr.class_id = class_id;
				dr.class_confidence = dr.confidence * best_cls_conf; 
				Box box((x + output_cpu[x_index]) * reciprocal_w,
					(y + output_cpu[y_index]) * reciprocal_h,
					exp(output_cpu[w_index]) * masked_anchors[n].width,
					exp(output_cpu[h_index]) * masked_anchors[n].height); //预测框 
				
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
void YoloModule::DeltaClass(float* output, float* delta, int cls_index, int class_id, float* p_class_conf) {

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
			else { //这是匹配的class_id
				if(p_class_conf) *p_class_conf += pred;
				delta[cls_index] = focal_grad * (1.0f - pred);
			}
		}
		else {

			if (c != class_id) {
				delta[cls_index] = -pred;
			}
			else { //这是匹配的class_id
				if (p_class_conf) *p_class_conf += pred;
				delta[cls_index] = 1.0f - pred;
			}

		} 
	}
}

 
bool YoloModule::Forward(ForwardContext& context) {
	if (!InferenceModule::Forward(context)) return false; 
	if (context.input->DataFormat() != CUDNN_TENSOR_NCHW) return false;
	int b, n;
	int classes = network->GetClassCount();
	int expected_channels = (classes + 5) * masked_anchors.size();
	if (input_channels != expected_channels) {
		cerr << " Error: Input channel count should be " << expected_channels << "! \n";
		return false;
	}


	if (context.input->DataType() == CUDNN_DATA_FLOAT) {
		output = *(context.input);
	}
	else {
		float *out = reinterpret_cast<float *>(output.Data());
		half *in = reinterpret_cast<__half*>(context.input->Data());
		if(!f16_to_f32( out, in, context.input->Elements())) return false;
	}
		
	float* output_gpu = reinterpret_cast<float*>(output.Data());
 
	bool ret = true; 
	int total_cells = output_height * output_width;
	int xy_elements = 2 * total_cells;
	int conf_cls_elements = (1 + classes) * total_cells;
	for (b = 0; b < output.Batch(); b++) {
		for (n = 0; n < (int)masked_anchors.size(); n++) {
			float* x = output_gpu + EntryIndex(n, 0, INDEX_PROBILITY_X);
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
	if (context.input->SameShape(shortcut_delta))
		shortcut_delta = 0.0f;
	else if (!shortcut_delta.Init(network->MiniBatch(), output_channels,output_height, output_width))
		return false;

	return CalcDelta();
}


float YoloModule::DeltaTruth(const TruthInLayer& truth, float* o,
	float* d, int cells, RotateType rt, float& object_conf, float& class_conf, int& class_identified) {
	//这个格子里有物体，并且由第anchor个anchor负责预测。
	float iou = 0.0f;
	int cell_index = truth.cell_y* output_width + truth.cell_x;
	int x_index = EntryIndex(truth.best_anchor_index, cell_index, INDEX_PROBILITY_X);
	int y_index = x_index + cells;
	int w_index = y_index + cells;
	int h_index = w_index + cells;
	int conf_index = h_index + cells;
	int cls_index = conf_index + cells;
	float anchor_w, anchor_h;
// 	if (rt == ToRight || rt == ToRight) {
// 		anchor_w = masked_anchors[truth.best_anchor_index].height * output_width;
// 		anchor_h = masked_anchors[truth.best_anchor_index].width * output_height;
// 	}
// 	else {
		anchor_w = masked_anchors[truth.best_anchor_index].width * output_width;
		anchor_h = masked_anchors[truth.best_anchor_index].height * output_height;
	//} 	
 
	float tx = truth.box.x - truth.cell_x; 
	float ty = truth.box.y - truth.cell_y ;
	 
	float tw = log(truth.box.w / anchor_w );
	float th = log(truth.box.h / anchor_h );
	float scale = (2.0f - truth.box.w * truth.box.h / cells) ;


	
	d[x_index] = /*scale **/ (tx - o[x_index]);
	d[y_index] = /*scale **/ (ty - o[y_index]);
	d[w_index] = scale * (tw - o[w_index]);
	d[h_index] = scale * (th - o[h_index]);
	object_conf = o[conf_index];
	d[conf_index] = (1.0f - o[conf_index]);


	Box pred_box(o[x_index] , o[y_index], exp(o[w_index])  , exp(o[h_index])  ); //预测框
	Box truth_box( tx, ty, truth.box.w / anchor_w, truth.box.h / anchor_h);
	iou = BoxIoU(pred_box, truth_box);
	float temp = 0.0f;

	DeltaClass(o, d, cls_index, truth.class_id, &temp);
	class_conf += temp;
	if (temp > 0.5f) class_identified++;
	return iou;
}
 
struct AnchorPredictTruth {
	
	int cell_x;
	int cell_y;
	int anchor_index;
	int truth_index;
};
#if 1
bool YoloModule::CalcDelta(){

	float loss = 0.0f;
	float recall_50 = 0.0f;		//正确识别，且IoU>=50%的数量
	float recall_75 = 0.0f;		//正确识别，且IoU>=75%的数量
	int false_positives = 0;		//预测错误的数量
	int true_positives = 0;		//预测正确的数量			
	float avr_iou = 0.0f;		//平均的IoU，不算预测错误的
	int recall_class = 0;	//正确识别，且类别也对的数量
	float avr_class_confidence = 0.0f;
	float avr_object_confidence = 0.0f; 
	int truth_count = 0;

	int offset = 0;
	CpuPtr<float> delta_cpu(output.Elements3D());
	CpuPtr<float> output_cpu(delta_cpu.Length());

	int classes = network->GetClassCount();
	 
	vector<TruthInLayer> truths; // for one batch 
	bool train_bg = GetAppConfig().TrainBackground();
 
	int total_cells = output_height * output_width;
	//CpuPtr<int> truth_in_cells(masked_anchors.size() * total_cells);
	for (int b = 0; b < output.Batch(); b++) {
		RotateType rt = network->GetRotateInfo(b);
		if (!output_cpu.Pull(output.BatchData(b))) return false;
		// 第一步：先确定哪些Gt是应该在这层预测的
		const LPObjectInfos all_truths = network->GetBatchTruths(b);
		if (!all_truths) return false;
		// 准备一个matrix， 
		delta_cpu.Reset(); 
		//truth_in_cells.Reset();
		Box anchor_box(0.0f,0.0f,0.0f,0.0f); 
		for (int t = 0; t < (int)all_truths->size(); t++) {
			const ObjectInfo& object = all_truths->at(t); 
			Box object_box(0,0,object.w, object.h);
			int best_anchor_index = -1;
			float best_iou = 0.0f;
			float best_anchor_width = 0.0f, best_anchor_height = 0.0f;
			// 找到最好的anchor
			for (int a = network->GetAnchorCount() - 1; a >= 0; a--) {				
				//if(rt == ToRight || rt == ToLeft)
				//	network->GetAnchor(a, anchor_box.h, anchor_box.w);
				//else
					network->GetAnchor(a, anchor_box.w, anchor_box.h);
				float iou = BoxIoU(object_box, anchor_box);
				if (iou > best_iou) { 
					best_anchor_index = a;
					best_iou = iou;
					best_anchor_width = anchor_box.w * output_width;
					best_anchor_height = anchor_box.h * output_height;
				}
			}
			//看看是不是在本层，是的话就放到layer_objects里面			
			for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {
				if (masked_anchors[a].masked_index == best_anchor_index) {
					//all_objects_in_layer++;
					TruthInLayer truth = { 0 };
					truth.box.x = object.x * output_width;
					truth.cell_x = floor(truth.box.x);

					truth.box.y = object.y * output_height; 
					truth.cell_y = floor(truth.box.y);

					truth.original = &object;

					truth.box.w = object.w * output_width;
					truth.box.h = object.h * output_height; 
					truth.best_anchor_index = a;
					truth.class_id = object.class_id; 
					truth.best_iou = best_iou; 
					truth.best_anchor_height = best_anchor_height;
					truth.best_anchor_width = best_anchor_width;
					truths.push_back(truth); 
					break;
				}
			}
			
		}
		float anchor_w, anchor_h, has_obj;
		int x_index, y_index, w_index, h_index, conf_index, cls_index;
		for (int y = 0, cell_index = 0, mat_index = 0; y < output_height; y++) {
			for (int x = 0; x < output_width; x++, cell_index++) { 
				for (int a = (int)masked_anchors.size() - 1; a >= 0; a --) {
					//GetPredictBox 
					AnchorBoxItem& anchor = masked_anchors[a];
					x_index = EntryIndex(a, cell_index, INDEX_PROBILITY_X);
					y_index = x_index + total_cells;
					w_index = y_index + total_cells;
					h_index = w_index + total_cells;
					conf_index = h_index + total_cells; 					

					int best_class_id = -1; 
					float best_cls_conf = 0.0f;
					cls_index = conf_index + total_cells;


					//先看看class_confidence，有没有大于0.5的 
					for (int c = 0, index = cls_index; c < classes; c++, index += cells_count) {
						if (output_cpu[index] > best_cls_conf) {
							best_class_id = c;
							best_cls_conf = output_cpu[index];
							break;
						}
					}
					has_obj = output_cpu[conf_index] * best_cls_conf;
					//if (has_obj < 0.5f) continue; //预测器认为这里没有GT，返回去
					
					// 觉得这里有GT，看看是不是有。最匹配的GT是哪个？

					/*if (rt == ToLeft || rt == ToRight) {
						anchor_w = anchor.height * output_width; 
						anchor_h = anchor.width * output_height;
					}
					else {*/
						anchor_w = anchor.width * output_width;
						anchor_h = anchor.height * output_height;
					//}
						int truth_index = -1;
						float best_iou = 0.0f;
						if (output_cpu[conf_index] > 0.5f) {
							Box pred_box(x + output_cpu[x_index], y + output_cpu[y_index],
								exp(output_cpu[w_index]) * anchor_w,
								exp(output_cpu[h_index]) * anchor_w);

							for (int t = (int)truths.size() - 1; t >= 0; t--) {
								TruthInLayer& truth = truths[t];
								float iou = BoxIoU(truth.box, pred_box);
								if (iou > 0.5f && iou > best_iou) {
									best_iou = iou;
									truth_index = t;
								}
							}
						}
					//最匹配的GT是哪个？
					if (truth_index != -1) {
						//召回了（会不会不是我们所期待的anchor？暂时不care）						
						TruthInLayer& truth = truths[truth_index]; 
						if(train_bg)
							delta_cpu[conf_index] = 1.0f - output_cpu[conf_index];
						// 但分类不一定对，检查对不对
						if (best_class_id != truth.class_id) { //分类没有预测对，对分类做delta
							
							false_positives++; //这个地方值得商榷
						}
						else {
							//分类和IoU都对了，记录一下召回
							if (0 == truth.recalls) true_positives++;
							truth.recalls++;
							if (best_iou > truth.best_iou) {
								truth.best_iou = best_iou;
								truth.best_recall_x = x;
								truth.best_recall_y = y;
								truth.best_recall_anchor = a;
							}
						}
						DeltaClass(output_cpu.ptr, delta_cpu.ptr, cls_index, truth.class_id);
					}
					else {
						//“认为有”目标，但预测出来的数据太难看，这种情况同样是我们不想要的
						// 干脆让delta_cpu[conf_index] = 0
						if (train_bg) {
							//delta_cpu[conf_index] = -output_cpu[conf_index];
							float prob = output_cpu[conf_index];
							float d = prob * focal_loss_delta((1.0f - prob), 0.1f, 4.0f);
							//float d =  focal_loss_delta((1.0f - prob), 0.5f, 4.0f);
							if (d > 0.001)
								delta_cpu[cls_index] = -d;
						}
						
						if(output_cpu[conf_index] > 0.5f)
							false_positives++;  
					}

				}
			}
		}
		//float loss1 = square_sum_array(delta_cpu, output.Elements3D());
		for (int t = (int)truths.size() - 1; t >= 0; t--) { 
			float iou = DeltaTruth(truths[t], output_cpu, delta_cpu, total_cells,rt,
				avr_object_confidence, avr_class_confidence, recall_class);
			if (iou >= 0.75f) {
				recall_75 += 1.0f;
				recall_50 += 1.0f;
			}
			else if (iou >= 0.5f) {
				recall_50 += 1.0f;
			}
			avr_iou += iou;
		} 
		truth_count += truths.size();
		truths.clear();
		shortcut_delta.Push(delta_cpu, offset, delta_cpu.Length());
		offset += delta_cpu.Length(); 
		loss += square_sum_array(delta_cpu, output.Elements3D());  
	}

	int total_recalls = true_positives + false_positives; 

	float prec = (total_recalls > 0) ? true_positives * 100.f / total_recalls : 0.0f; 
	loss = (truth_count == 0) ? 0.0f : (loss / truth_count);
	network->RegisterTrainingResults(loss,  true_positives, false_positives);
 
	float reca = (truth_count > 0 ) ? (true_positives * 100.0f) / truth_count : 0.0f;
	char line[300];
	sprintf(line," %s: %d/%d objects found. Recall: %.2f%%, Precision: %.2f%%, %d class identified, loss: %.4f.\n",
		name.c_str(), true_positives, truth_count,  reca, prec, recall_class, loss);
	cout << line;
	if (truth_count > 0) {
		avr_iou /= truth_count;
		recall_50 /= truth_count;
		recall_75 /= truth_count;
		avr_class_confidence /= truth_count;
		avr_object_confidence /= truth_count;
		sprintf(line, "          IoU: %.6f, .5R: %.2f%%, .75R: %.2f%%, Obj Conf: %.2f%%, Class Conf: %.2f%%.\n",avr_iou, recall_50 * 100.0f,
			recall_75 * 100.0f, avr_object_confidence * 100.0f, avr_class_confidence * 100.0f);
		cout << line;
	} 
	 
	return true;
} 
#else
bool YoloModule::CalcDelta() {
	float loss = 0.0f;
	float recall_50 = 0.0f;		//正确识别，且IoU>=50%的数量
	float recall_75 = 0.0f;		//正确识别，且IoU>=75%的数量
	int recall_err = 0;		//预测错误的数量
	int num_recalls = 0;		//预测正确的数量			
	float avr_iou = 0.0f;		//平均的IoU，不算预测错误的
	int recall_class = 0;	//正确识别，且类别也对的数量
	float avr_class_confidence = 0.0f;
	float avr_object_confidence = 0.0f;
	int miss_truth_count = 0;
	int truth_count = 0;

	int offset = 0;
	CpuPtr<float> delta_cpu(output.Elements3D());
	CpuPtr<float> output_cpu(delta_cpu.Length());

	vector<TruthInLayer> truths; // for one batch  

	int total_cells = output_height * output_width;
	


	float scale_min, scale_max; 
	vector<float> box_ratios;
	switch (network->GetInputWidth() / output_width) {
	case 32:
		scale_min = 0.0f;
		scale_max = 0.04f;
		box_ratios = { 2.0 / 3.0, 1.0f, 1.5f };
		break;
	case 16:
		scale_min = 0.04f;
		scale_max = 0.25f;
		box_ratios = { 0.25f ,0.5f, 0.75f, 1.0f};
		break;
	default:
		scale_min = 0.25f;
		scale_max = 1.0f;
		box_ratios = { 1.0f/3.0f, 0.5f, 2.0f/3.0f, 1.0f, 1.5f };
		break;
	}

	for (int b = 0; b < output.Batch(); b++) {
		RotateType rt = network->GetRotateInfo(b);
		if (!output_cpu.Pull(output.BatchData(b))) return false;
		// 第一步：先确定哪些Gt是应该在这层预测的
		const LPObjectInfos all_truths = network->GetBatchTruths(b);
		if (!all_truths) return false;
		// 准备一个matrix， 
		delta_cpu.Reset(); 
		Box anchor_box(0.0f, 0.0f, 0.0f, 0.0f);
		float tx, ty;
		for (int t = 0; t < (int)all_truths->size(); t++) {
			const ObjectInfo& truth = all_truths->at(t);
			float temp = truth.h * truth.h + truth.w * truth.w;
			if (temp > scale_min && temp < scale_max) {
				TruthInLayer  layerT = { 0 };
				layerT.orig_truth = &truth;
				tx = truth.x * output_width;
				ty = truth.y * output_height;
				layerT.cell_x = floor(tx);
				layerT.cell_y = floor(ty);
				layerT.x_offset = tx - layerT.cell_x - 0.5f;
				layerT.y_offset = ty - layerT.cell_y - 0.5f;
				layerT.class_id = truth.class_id;
				truths.push_back(layerT);
			}
		}
		cout << name << ": " << truths.size() << " GT to predict.";
		int x_index, y_index, w_index, h_index, conf_index, cls_index;
		for (int y = 0, cell_index = 0, mat_index = 0; y < output_height; y++) {
			for (int x = 0; x < output_width; x++, cell_index++) {
				for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {
					//GetPredictBox 
					AnchorBoxItem& anchor = masked_anchors[a];
					x_index = EntryIndex(a, cell_index, INDEX_PROBILITY_X);
					y_index = x_index + total_cells;
					w_index = y_index + total_cells;
					h_index = w_index + total_cells;
					conf_index = h_index + total_cells;
					cls_index = conf_index + total_cells;
					float tempw = exp(output_cpu[w_index]);
					float temph = exp(output_cpu[h_index]);
					if (rt == ToLeft || rt == ToRight) {
						tempw *= anchor.height;
						temph *= anchor.width;
					}
					else {
						tempw *= anchor.width;
						temph *= anchor.height;
					}

					Box pred_box(x + output_cpu[x_index], y + output_cpu[y_index],
						tempw * output_width, temph * output_height);
					int truth_index = -1;
					float best_iou = 0.0f;
					for (int t = (int)truths.size() - 1; t >= 0; t--) {
						TruthInLayer& truth = truths[t];
						/*if (truth.cur_pred_x == x &&
						truth.cur_pred_y == y && truth.cur_anchor_index == a) {
						cout << " *** Find the predictor\n";
						}*/
						float iou = BoxIoU(truth.box, pred_box);
						if (iou > best_iou) {
							best_iou = iou;
							truth_index = t;
						}
					}
					if (best_iou > truth_thresh) {
						TruthInLayer& truth = truths[truth_index];
						if (0 == truth.recalls) num_recalls++;
						truth.recalls++;
						delta_cpu[conf_index] = 0.0f; //认为识别正确
					}
					else {
						delta_cpu[conf_index] = -output_cpu[conf_index];// *100;
					}
					if (best_iou < ignore_thresh && output_cpu[conf_index] > ignore_thresh) {
						recall_err++;
						//要不要加大剂量？
					}

				}
			}
		}
		//float loss1 = square_sum_array(delta_cpu, output.Elements3D());
		for (int t = (int)truths.size() - 1; t >= 0; t--) {
			//cout << " +++ delta for truth " << dec << t << "... ";
			float iou = DeltaTruth(truths[t], output_cpu, delta_cpu, total_cells, rt,
				avr_object_confidence, avr_class_confidence, recall_class);
			if (iou >= 0.75f) {
				recall_75 += 1.0f;
				recall_50 += 1.0f;
			}
			else if (iou >= 0.5f) {
				recall_50 += 1.0f;
			}
			avr_iou += iou;
		}
		truth_count += truths.size();
		truths.clear();
		miss_truth_count += miss_truths.size();
		miss_truths.clear();
		shortcut_delta.Push(delta_cpu, offset, delta_cpu.Length());
		offset += delta_cpu.Length();
		loss += square_sum_array(delta_cpu, output.Elements3D());
	}

	int total_recalls = num_recalls + recall_err;

	float prec = (total_recalls > 0) ? num_recalls * 100.f / total_recalls : 0.0f;
	loss = (truth_count == 0) ? 0.0f : (loss / truth_count);
	network->RegisterTrainingResults(loss, miss_truth_count, num_recalls, recall_err);

	float reca = (truth_count > 0) ? (num_recalls * 100.0f) / truth_count : 0.0f;
	char line[300];
	sprintf(line, " %s: %d/%d objects found (%d unpredictable). Recall: %.2f%%, Precision: %.2f%%, %d class identified, loss: %.4f.\n",
		name.c_str(), num_recalls, truth_count, miss_truth_count, reca, prec, recall_class, loss);
	cout << line;
	if (truth_count > 0) {
		avr_iou /= truth_count;
		recall_50 /= truth_count;
		recall_75 /= truth_count;
		avr_class_confidence /= truth_count;
		avr_object_confidence /= truth_count;
		sprintf(line, "          IoU: %.6f, .5R: %.2f%%, .75R: %.2f%%, Obj Conf: %.2f%%, Class Conf: %.2f%%.\n", avr_iou, recall_50 * 100.0f,
			recall_75 * 100.0f, avr_object_confidence * 100.0f, avr_class_confidence * 100.0f);
		cout << line;
	}

	return true;
}
 
bool YoloModule::RescueMissTruth(TruthInLayer & missT, CpuPtr<int>& truth_map, int miss_truth_index, RotateType rt) {
	int start_x =  missT.cell_x - missT.box.w  - 1;
	int stop_x = missT.cell_x + missT.box.w + 1;
	int start_y = missT.cell_y - missT.box.h - 1;
	int stop_y = missT.cell_y + missT.box.h + 1;
	if (start_x < 0) start_x = 0;
	if (start_y < 0) start_y = 0;
	if (stop_x >= output_width)
		stop_x = output_width - 1;
	if (stop_y >= output_height)
		stop_y = output_height - 1;
	//int start_y = (missT.center_y < 0.5) ? missT.cell_y - 1 : missT.cell_y;
	int best_x = -1;
	int best_y = -1;
	int best_iou = 0.0f;
	int best_anchor = -1;
	Box anchor_box;
	for (int y =  start_y; y <= stop_y; y++) {
		for (int x = start_x; x <= stop_x; x++) {
			for (int a = (int)masked_anchors.size() - 1; a >= 0; a--) {
				int i = (y * output_width + x) *  (int)masked_anchors.size() + a;
				if (truth_map[i]) continue;//这个anchor已经有truth了
				if (x < missT.cell_x)
					anchor_box.x = x * output_width;
				else if (x > missT.cell_x)
					anchor_box.x = (x + 1) * output_width;
				else
					anchor_box.x = missT.box.x;
				if (y < missT.cell_y)
					anchor_box.y = y * output_height;
				else if (y > missT.cell_y)
					anchor_box.y = (y + 1) * output_height;
				else
					anchor_box.y = missT.box.y;
				if (rt == ToRight || rt == ToLeft) {
					anchor_box.w = masked_anchors[a].height * output_width;
					anchor_box.h = masked_anchors[a].width * output_height;
				}
				else {
					anchor_box.w = masked_anchors[a].width * output_width;
					anchor_box.h = masked_anchors[a].height * output_height;
				}
				float iou = BoxIoU(missT.box, anchor_box);
				if ( iou > ignore_thresh && iou > best_iou) { //先这样吧
					best_iou = iou;
					best_x = x;
					best_y = y;
					best_anchor = a;
				}
			}
		}
	}
	if (-1 == best_anchor) return false; //没有合适的anchor，这基本不存在
	if ( best_iou > ignore_thresh) {
		int map_index = (best_y * output_width + best_x) * (int)masked_anchors.size() + best_anchor;
		//行，给它找到一个另外的anchor——“抢救”回来了
		truth_map[map_index] = miss_truth_index;
		missT.cur_pred_x = best_x;
		missT.cur_pred_y = best_y;
		missT.cur_anchor_index = best_anchor;
		missT.best_iou = best_iou;
		return true;
	}
	return false;
}
#endif
bool YoloModule::Backward(CudaTensor& delta) {
	delta = shortcut_delta; 
	return shortcut_delta.Release();
}
// There is a bug in NCS2, not using the "RegionYolo" layer is a workaround.
bool YoloModule::OutputIRModel(ofstream& xml, ofstream& bin, stringstream& edges, size_t& bin_offset, int& l_index) const {
#if 0
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
#endif
	return true;
}
uint32_t YoloModule::GetFlops() const {
	return 0;
}