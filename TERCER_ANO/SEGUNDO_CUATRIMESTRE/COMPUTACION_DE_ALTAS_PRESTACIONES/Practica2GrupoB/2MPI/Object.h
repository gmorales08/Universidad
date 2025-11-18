#pragma once

#include "Shape.h"
#include "Material.h"

class Object {
public:
	Object(Shape* shape, Material* material) : s(shape), m(material) {}

	bool checkCollision(const Ray& ray, float t_min, float t_max, CollisionData& cd) {
		return (s->collide(ray, t_min, t_max, cd));
	}

	bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) {
		return (m->scatter(ray, cd, attenuation, scattered));
	}

private:
	Shape* s;
	Material* m;
};
