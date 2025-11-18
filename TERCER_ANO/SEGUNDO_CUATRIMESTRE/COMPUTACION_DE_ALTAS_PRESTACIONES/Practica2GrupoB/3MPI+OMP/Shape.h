#pragma once

#include "Ray.h"
#include "CollisionData.h"

class Shape {
public:
	virtual bool collide(const Ray& ray, float t_min, float t_max, CollisionData& cd) const = 0;
};
