#pragma once
struct ObjectInfo;
class Box {
public:
	float x;
	float y;
	float w;
	float h;
	Box(float center, float middle, float width, float height);
	Box(const float* var = NULL);
	Box(const ObjectInfo& item);
	inline float Area() const { return w * h; }
};
float BoxIntersection(const Box& A, const Box& B);
float BoxUnion(const Box& A, const Box& B);
float BoxIoU(const Box& A, const Box& B);
float BoxRMSE(const Box& A, const Box& B);
