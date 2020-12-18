#include "stdafx.h"
#include "network.h"
#include "box.h"
#include "yolo.h"

Box::Box(float center, float middle, float width, float height) {
	x = center;
	y = middle;
	w = width;
	h = height;
}
Box::Box(const float* var) {
	if (var) {
		x = var[0];
		y = var[1];
		w = var[2];
		h = var[3];
	}
	else
		x = y = w = h = 0.0;
}
Box::Box(const ObjectInfo& item) {
	x = item.x;
	y = item.y;
	w = item.w;
	h = item.h;
}
static float overlap(float x1, float w1, float x2, float w2) {
	float l1 = x1 - w1 * 0.5;
	float l2 = x2 - w2  * 0.5;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1  * 0.5;
	float r2 = x2 + w2  * 0.5;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}
float BoxIntersection(const Box& A, const Box& B) {
	float w = overlap(A.x, A.w, B.x, B.w);
	float h = overlap(A.y, A.h, B.y, B.h);
	if (w < 0 || h < 0) return 0.0;

	return w * h;
}
float BoxUnion(const Box& A, const Box& B) {
	float i = BoxIntersection(A, B);
	return A.Area() + B.Area() - i;
}
float BoxIoU(const Box& A, const Box& B) {
	float i = BoxIntersection(A, B);
	if (0.0 == i) return 0.0;
	float u = A.Area() + B.Area() - i;
	if (0.0 == u) {
		cerr << "Error: zero divisor in BoxIoU!\n";
		return 1.0;
	}
	return i / u;
}
Box MinContainer(const Box& a, const Box& b) {
    float le = fmin(a.Left(), b.Left());
    float ri = fmax(a.Right(), b.Right());
    float to = fmin(a.Top(), b.Top());
    float bo = fmax(a.Bottom(), b.Bottom());

    Box c(0.5f * (le + ri),0.5f * (to + bo), ri - le, bo - to);
    return c;
} 
float BoxDIoU(const Box& a, const Box& b) {
    Box ab = MinContainer(a, b);

    float iou = BoxIoU(a, b);
    float c_2 = ab.w * ab.w + ab.h * ab.h;
    if (0.0f == c_2) return iou;

    float d_x = a.x - b.x;
    float d_y = a.y - b.y;

    float p_2 = d_x * d_x + d_y * d_y; 

    return iou - p_2 / c_2  ;
}
float BoxCIoU(const Box& a, const Box& b ) { 
    Box ab = MinContainer(a, b);

    float iou = BoxIoU(a, b);
    float c_2 = ab.w * ab.w + ab.h * ab.h;
    if (0.0f == c_2) return iou;

    float d_x = a.x - b.x;
    float d_y = a.y - b.y;

    float p_2 = d_x * d_x + d_y * d_y;

    float v_sqrt = 2.0f * (atan2(b.w, b.h) - atan2(a.w, a.h)) / M_PI;

    float v = v_sqrt * v_sqrt;

    float alpha = v / (1.0f - iou + v + 0.0000001f);

    return iou - p_2 / c_2 - alpha * v;

} 
float DeltaCIoU(const Box& pred, const Box& gt, float& delta_x, float& delta_y, float& delta_w, float& delta_h) {


    float pred_t = pred.Top(), pred_b = pred.Bottom(), pred_l = pred.Left(), pred_r = pred.Right(); 
    float gt_t = gt.Top(), gt_b = gt.Bottom(), gt_l = gt.Left(), gt_r = gt.Right();

    float X = pred.Area(), Xhat = gt.Area();

    float Ih = fmin(pred_b, gt_b) - fmax(pred_t, gt_t);
    float Iw = fmin(pred_r, gt_r) - fmax(pred_l, gt_l);
    float I = Iw * Ih;
    float U = X + Xhat - I;

    float iou = I / U;
    float x_diff = pred.x - gt.x, y_diff = pred.y - gt.y;
    float S = x_diff * x_diff + y_diff * y_diff;
    float giou_Cw = fmax(pred_r, gt_r) - fmin(pred_l, gt_l);
    float giou_Ch = fmax(pred_b, gt_b) - fmin(pred_t, gt_t);
    float giou_C = giou_Cw * giou_Ch;
 
    float dX_wrt_t = -pred.w;
    float dX_wrt_b = pred.w;
    float dX_wrt_l = - pred.h;
    float dX_wrt_r = pred.h;
    

    // gradient of I min/max in IoU calc (prediction)
    float dI_wrt_t = pred_t > gt_t ? (-Iw) : 0;
    float dI_wrt_b = pred_b < gt_b ? Iw : 0;
    float dI_wrt_l = pred_l > gt_l ? (-Ih) : 0;
    float dI_wrt_r = pred_r < gt_r ? Ih : 0;
    // derivative of U with regard to x
    float dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float dC_wrt_t = pred_t < gt_t ? (-1 * giou_Cw) : 0;
    float dC_wrt_b = pred_b > gt_b ? giou_Cw : 0;
    float dC_wrt_l = pred_l < gt_l ? (-1 * giou_Ch) : 0;
    float dC_wrt_r = pred_r > gt_r ? giou_Ch : 0;

    float p_dt = 0, p_db = 0, p_dl = 0, p_dr = 0;
    if (U > 0) {
        p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
        p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
        p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
        p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }
    // apply grad from prediction min/max for correct corner selection
   
    Box box_c = MinContainer(pred, gt);
    float Ct = box_c.Top(),Cb = box_c.Bottom(), Cl = box_c.Left(), Cr = box_c.Right();
    //float box_c.w = Cr - Cl;
   // float box_c.h = Cb - Ct;
    float C = box_c.w * box_c.w + box_c.h * box_c.h;

    float dCt_dx = 0;
    float dCt_dy = pred_t < gt_t ? 1 : 0;
    float dCt_dw = 0;
    float dCt_dh = pred_t < gt_t ? -0.5 : 0;

    float dCb_dx = 0;
    float dCb_dy = pred_b > gt_b ? 1 : 0;
    float dCb_dw = 0;
    float dCb_dh = pred_b > gt_b ? 0.5 : 0;

    float dCl_dx = pred_l < gt_l ? 1 : 0;
    float dCl_dy = 0;
    float dCl_dw = pred_l < gt_l ? -0.5 : 0;
    float dCl_dh = 0;

    float dCr_dx = pred_r > gt_r ? 1 : 0;
    float dCr_dy = 0;
    float dCr_dw = pred_r > gt_r ? 0.5 : 0;
    float dCr_dh = 0;

    float dCw_dx = dCr_dx - dCl_dx;
    float dCw_dy = dCr_dy - dCl_dy;
    float dCw_dw = dCr_dw - dCl_dw;
    float dCw_dh = dCr_dh - dCl_dh;

    float dCh_dx = dCb_dx - dCt_dx;
    float dCh_dy = dCb_dy - dCt_dy;
    float dCh_dw = dCb_dw - dCt_dw;
    float dCh_dh = dCb_dh - dCt_dh;


    delta_x = p_dl + p_dr;           //delta_x, delta_y, delta_w and delta_h are the gradient of IoU or GIoU.
    delta_y = p_dt + p_db;
    delta_w = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
    delta_h = (p_db - p_dt);

     
    
    float atan_diff = atan2(gt.w, gt.h) - atan2(pred.w, pred.h);
    float temp = 1.0f / (M_PI * M_PI);
    float v = 4 * temp  * atan_diff * atan_diff;
    float alpha = v / (1.0f - iou + v + 0.0000001f);
    float ar_dw = 8 * temp * atan_diff * pred.h;
    float ar_dh = -8 * temp * atan_diff * pred.w;
    temp = 1.0f / (C * C);
    if (C > 0) {
        if (Iw <= 0 || Ih <= 0) {
            delta_x = (2 * (gt.x - pred.x) * C - (2 * box_c.w * dCw_dx + 2 * box_c.h * dCh_dx) * S) * temp;
            delta_y = (2 * (gt.y - pred.y) * C - (2 * box_c.w * dCw_dy + 2 * box_c.h * dCh_dy) * S) * temp;
            delta_w = (2 * box_c.w * dCw_dw + 2 * box_c.h * dCh_dw) * S * temp + alpha * ar_dw;
            delta_h = (2 * box_c.w * dCw_dh + 2 * box_c.h * dCh_dh) * S * temp + alpha * ar_dh;
        }
        else {
            // dar*
            delta_x += (2 * (gt.x - pred.x) * C - (2 * box_c.w * dCw_dx + 2 * box_c.h * dCh_dx) * S) * temp;
            delta_y += (2 * (gt.y - pred.y) * C - (2 * box_c.w * dCw_dy + 2 * box_c.h * dCh_dy) * S) * temp;
            delta_w += (2 * box_c.w * dCw_dw + 2 * box_c.h * dCh_dw) * S * temp + alpha * ar_dw;
            delta_h += (2 * box_c.w * dCw_dh + 2 * box_c.h * dCh_dh) * S * temp + alpha * ar_dh;
        }
    }
    
    return 1.0f - iou + S / C + alpha * v;
}