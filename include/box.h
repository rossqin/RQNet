#pragma once
struct ObjectInfo;
class Box {
public:
	float x;
	float y;
	float w;
	float h;
	Box(float center, float middle, float width, float height);
	Box(const float* var = nullptr);
	Box(const ObjectInfo& item);
	inline float Area() const { return w * h; }
	inline float Left() const { return x - 0.5f * w; }
	inline float Right() const { return x + 0.5f * w; }
	inline float Top() const { return y - 0.5f * h; }
	inline float Bottom() const { return y + 0.5f * h; }
};
float BoxIntersection(const Box& a, const Box& b);
float BoxUnion(const Box& a, const Box& b);
float BoxIoU(const Box& a, const Box& b);
float BoxDIoU(const Box& a, const Box& b);
float BoxCIoU(const Box& a, const Box& b);
Box MinContainer(const Box& a, const Box& b);
float DeltaCIoU(const Box& pred, const Box& gt, float& delta_x, float& delta_y, float& delta_w, float& delta_h);