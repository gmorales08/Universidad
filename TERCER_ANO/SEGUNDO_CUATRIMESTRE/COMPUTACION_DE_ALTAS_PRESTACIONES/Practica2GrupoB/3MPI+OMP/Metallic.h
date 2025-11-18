#pragma once

#include "utils.h"
#include "random.h"

#include "Material.h"

class Metallic : public Material {
public:
	Metallic(const Vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

	bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const;

private:
	Vec3 albedo;
	float fuzz;
};
