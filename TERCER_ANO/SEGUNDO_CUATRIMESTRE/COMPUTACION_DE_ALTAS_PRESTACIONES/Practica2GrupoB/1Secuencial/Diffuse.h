#pragma once

#include "Vec3.h"
#include "Material.h"

class Diffuse : public Material {
public:
	Diffuse(const Vec3& color) : color(color) {}

	bool scatter(const Ray& ray, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const {
		Vec3 target = cd.p + cd.normal + randomNormalSphere();
		scattered = Ray(cd.p, target - cd.p);
		attenuation = color;
		return true;
	}
private:
	Vec3 color;
};
