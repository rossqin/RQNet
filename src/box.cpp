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
float BoxRMSE(const Box& A, const Box& B) {
	float dx = A.x - B.x;
	float dy = A.y - B.y;
	float dw = A.w - B.w;
	float dh = A.h - B.h;

	return sqrt(dx * dx + dy * dy + dw * dw + dh * dh);

}