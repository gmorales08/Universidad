#pragma once

#include <vector>

#include "Object.h"

class Scene {
public:
	Scene(int depth = 50) : ol(), sky(), inf(), d(depth) {}
	Scene(const Scene& list) = default;

	void add(Object* h) { ol.push_back(h); }
	void setSkyColor(Vec3 sky) { this->sky = sky; }
	void setInfColor(Vec3 inf) { this->inf = inf; }

	Vec3 getSceneColor(const Ray& r);

protected:
	Vec3 getSceneColor(const Ray& r, int depth);

private:
	std::vector<Object*> ol;
	Vec3 sky;
	Vec3 inf;
	int d;
};
